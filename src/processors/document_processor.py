from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
import json
from datetime import datetime
import cv2
import numpy as np
import pdf2image
import os
import torch

from .ocr_processor import OCRProcessor
from .table_detector import TableDetector
from .field_extractor import FieldExtractor
from ..layoutlm.tokenization.layoutlm_tokenizer import LayoutLMTokenizer
from ..utils.logger import setup_logger

class DocumentProcessor:
    def __init__(self, 
                 config_path: str = "config/config.json",
                 model_path: str = "microsoft/layoutlm-base-uncased"):
        """Initialize the document processor."""
        self.logger = setup_logger(__name__)
        
        # Load config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Initialize components
        self.ocr_processor = OCRProcessor()
        self.table_detector = TableDetector()
        self.field_extractor = FieldExtractor(config_path)
        
        # Load tokenizer
        self.logger.info(f"Loading tokenizer from: {model_path}")
        self.tokenizer = LayoutLMTokenizer(model_path)
        self.logger.info("Tokenizer loaded successfully.")

    def process_document(self, 
                        file_path: Union[str, Path],
                        output_dir: Optional[str] = None) -> Dict:
        """Process document and generate LayoutLM-compatible dataset."""
        try:
            # Validate input file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Document not found: {file_path}")
                
            self.logger.info(f"Processing document: {file_path}")
            
            # Convert PDF to images if needed
            if str(file_path).lower().endswith('.pdf'):
                images = self._convert_pdf_to_images(file_path)
            else:
                images = [self._load_image(file_path)]
            
            # Process each page
            all_results = []
            for page_num, image in enumerate(images, 1):
                self.logger.info(f"Processing page {page_num}")
                page_results = self._process_page(image, page_num)
                all_results.append(page_results)
            
            # Combine results from all pages
            combined_results = self._combine_page_results(all_results)
            
            # Add metadata
            combined_results['metadata'] = self._create_metadata(file_path)
            
            # Validate results
            self._validate_results(combined_results)
            
            # Save results if output directory provided
            if output_dir:
                self._save_results(combined_results, file_path, output_dir)
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {str(e)}")
            raise

    def _convert_pdf_to_images(self, pdf_path: Union[str, Path]) -> List[np.ndarray]:
        """Convert PDF to list of images."""
        try:
            # Convert PDF pages to PIL Images
            pil_images = pdf2image.convert_from_path(pdf_path)
            
            # Convert PIL Images to numpy arrays
            return [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in pil_images]
            
        except Exception as e:
            self.logger.error(f"Error converting PDF to images: {str(e)}")
            raise

    def _load_image(self, file_path: Union[str, Path]) -> np.ndarray:
        """Load image file to numpy array."""
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError(f"Could not load image: {file_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _process_page(self, image: np.ndarray, page_num: int) -> Dict:
        """Process a single page of the document."""
        try:
            # Extract text and boxes with OCR
            ocr_results = self.ocr_processor.process_image(image)
            if not ocr_results['text']:
                self.logger.warning(f"No text detected on page {page_num}")
                return self._create_empty_page_result()
            
            # Add image size to OCR results
            ocr_results['image_size'] = image.shape[:2]  # (height, width)
            
            # Detect tables
            tables = self.table_detector.detect_tables(image, ocr_results)
            
            # Extract fields
            fields = self.field_extractor.extract_fields(ocr_results)
            
            # Process for LayoutLM
            layoutlm_features = self._prepare_layoutlm_features(
                ocr_results, 
                fields, 
                tables
            )
            
            return {
                'ocr_results': ocr_results,
                'extracted_fields': fields,
                'tables': tables,
                'layoutlm_features': layoutlm_features,
                'page_number': page_num
            }
            
        except Exception as e:
            self.logger.error(f"Error processing page {page_num}: {str(e)}")
            raise

    def _create_empty_page_result(self) -> Dict:
        """Create empty result for pages with no text."""
        return {
            'ocr_results': {
                'text': [],
                'boxes': [],
                'confidences': [],
                'image_size': None
            },
            'extracted_fields': {},
            'tables': [],
            'layoutlm_features': None,
            'page_number': 0
        }

    def _generate_iob_labels(self,
                           ocr_results: Dict,
                           fields: Dict,
                           tables: List) -> List[str]:
        """Generate IOB (Inside, Outside, Beginning) labels for tokens."""
        if not ocr_results or 'text' not in ocr_results or 'boxes' not in ocr_results:
            self.logger.warning("Invalid OCR results for label generation")
            return ["O"] * len(ocr_results.get('text', []))

        labels = ["O"] * len(ocr_results['text'])
        
        # Process fields
        for field_name, field_data in fields.items():
            if not field_data or 'value' not in field_data:
                continue
                
            for i, (word, box) in enumerate(zip(ocr_results['text'], ocr_results['boxes'])):
                # Skip empty words
                if not word.strip():
                    continue
                    
                # Check if word is part of field value
                if (field_data.get('value') and 
                    word.lower() in field_data['value'].lower() and 
                    self._is_within_box(box, field_data.get('bbox'))):
                    
                    # Determine if this is the start of a field
                    is_beginning = (i == 0 or labels[i-1] != f"I-{field_name.upper()}")
                    labels[i] = f"{'B' if is_beginning else 'I'}-{field_name.upper()}"
        
        # Process tables
        for table in tables:
            for cell in table.get('cells', []):
                if not cell or 'text' not in cell:
                    continue
                    
                cell_text = cell.get('text', '').lower()
                cell_bbox = cell.get('bbox')
                
                for i, (word, box) in enumerate(zip(ocr_results['text'], ocr_results['boxes'])):
                    if (word.lower() in cell_text and 
                        self._is_within_box(box, cell_bbox)):
                        
                        label_type = "TABLE_HEADER" if cell.get('is_header') else "TABLE_CELL"
                        is_beginning = (i == 0 or not labels[i-1].endswith(label_type))
                        labels[i] = f"{'B' if is_beginning else 'I'}-{label_type}"
        
        return labels

    def _is_within_box(self, box: List[int], target_box: Optional[List[int]]) -> bool:
        """Check if a box is within target box boundaries."""
        if not box or not target_box:
            return False
            
        try:
            return (box[0] >= target_box[0] and 
                    box[1] >= target_box[1] and 
                    box[2] <= target_box[2] and 
                    box[3] <= target_box[3])
        except (IndexError, TypeError):
            return False

    def _prepare_layoutlm_features(self,
                                 ocr_results: Dict,
                                 fields: Dict,
                                 tables: List) -> Dict:
        """Prepare features for LayoutLM."""
        try:
            # Validate OCR results
            if not ocr_results or 'text' not in ocr_results or 'boxes' not in ocr_results:
                raise ValueError("Invalid OCR results")
                
            # Generate IOB labels
            labels = self._generate_iob_labels(ocr_results, fields, tables)
            
            # Ensure boxes are in the correct format
            boxes = [[
                min(max(0, int(box[0])), 1000),
                min(max(0, int(box[1])), 1000),
                min(max(0, int(box[2])), 1000),
                min(max(0, int(box[3])), 1000)
            ] for box in ocr_results['boxes']]
            
            # Tokenize
            encoding = self.tokenizer.tokenize(
                words=ocr_results['text'],
                boxes=boxes,
                labels=labels
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze().tolist(),
                'attention_mask': encoding['attention_mask'].squeeze().tolist(),
                'token_type_ids': encoding['token_type_ids'].squeeze().tolist(),
                'bbox': encoding['bbox'].squeeze().tolist(),
                'labels': encoding['labels'].squeeze().tolist() if 'labels' in encoding else None
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing LayoutLM features: {str(e)}")
            raise

    def _combine_page_results(self, page_results: List[Dict]) -> Dict:
        """Combine results from multiple pages."""
        if not page_results:
            return self._create_empty_page_result()
            
        combined = {
            "ocr_results": {
                "text": [],
                "boxes": [],
                "confidences": [],
                "image_size": None
            },
            "extracted_fields": {},
            "tables": [],
            "layoutlm_features": None
        }
        
        for page in page_results:
            # Combine OCR results
            combined["ocr_results"]["text"].extend(page["ocr_results"]["text"])
            combined["ocr_results"]["boxes"].extend(page["ocr_results"]["boxes"])
            combined["ocr_results"]["confidences"].extend(page["ocr_results"]["confidences"])
            
            # Combine fields (take highest confidence values)
            for field_name, field_data in page["extracted_fields"].items():
                if (field_name not in combined["extracted_fields"] or
                    field_data.get('confidence', 0) > combined["extracted_fields"][field_name].get('confidence', 0)):
                    combined["extracted_fields"][field_name] = field_data
            
            # Combine tables
            combined["tables"].extend(page["tables"])
        
        # Use features from first page for now
        if page_results:
            combined["layoutlm_features"] = page_results[0]["layoutlm_features"]
            combined["ocr_results"]["image_size"] = page_results[0]["ocr_results"]["image_size"]
        
        return combined

    def _validate_results(self, results: Dict) -> None:
        """Validate processing results."""
        if not results:
            raise ValueError("Empty processing results")
            
        # Check for required fields
        required_fields = [
            field_name for field_name, field_config in self.config.get('fields', {}).items()
            if field_config.get('required', False)
        ]
        
        extracted_fields = results.get('extracted_fields', {})
        missing_fields = [
            field for field in required_fields
            if field not in extracted_fields or not extracted_fields[field].get('value')
        ]
        
        if missing_fields:
            error_handling = self.config.get('validation', {}).get('error_handling', {})
            if error_handling.get('missing_fields') == 'error':
                raise ValueError(f"Missing required fields: {missing_fields}")
            else:
                self.logger.warning(f"Missing required fields: {missing_fields}")

    def _create_metadata(self, file_path: Union[str, Path]) -> Dict:
        """Create metadata for processed document."""
        return {
            "source_file": str(file_path),
            "processing_date": datetime.now().isoformat(),
            "processor_version": "1.0.0",
            "model_name": self.tokenizer.tokenizer.name_or_path,
            "config_path": str(self.config)
        }

    def _save_results(self,
                     results: Dict,
                     file_path: Union[str, Path],
                     output_dir: str):
        """Save processing results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"processed_{Path(file_path).stem}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {output_file}")
