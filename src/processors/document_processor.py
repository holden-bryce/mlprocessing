import logging
import os
from typing import Dict
from datetime import datetime
import json
import torch
from .ocr_processor import OCRProcessor
from .field_extractor import FieldExtractor
from .table_detector import TableDetector
from .feature_generator import FeatureGenerator

class DocumentProcessor:
    def __init__(
        self,
        ocr_processor: OCRProcessor,
        field_extractor: FieldExtractor,
        table_detector: TableDetector,
        feature_generator: FeatureGenerator,
        output_dir: str
    ):
        """
        Initialize DocumentProcessor with all necessary components
        """
        # Set up model path
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = os.path.join(self.project_root, 'models', 'layoutlm-base-uncased')
        
        # Initialize components
        self.ocr_processor = ocr_processor
        self.field_extractor = field_extractor
        self.table_detector = table_detector
        self.feature_generator = feature_generator
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Ensure model directory exists
        os.makedirs(self.model_path, exist_ok=True)
        
        # Create document processing log directory
        self.doc_log_dir = os.path.join(output_dir, 'document_logs')
        os.makedirs(self.doc_log_dir, exist_ok=True)
        
        # Validate model path
        self._validate_model_path()

    def _validate_model_path(self):
        """
        Validate that required model files exist
        """
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json', 'vocab.txt']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(self.model_path, f))]
        
        if missing_files:
            self.logger.warning(f"Missing model files: {missing_files}")
            self.logger.info("Model files will be downloaded if needed by components")

    def process_document(self, pdf_path: str) -> Dict:
        """
        Process a document through all stages
        """
        try:
            # Start timing
            start_time = datetime.now()
            
            # Process document stages
            self.logger.info(f"Processing document: {pdf_path}")
            
            # 1. OCR Processing
            self.logger.info("Starting OCR processing...")
            ocr_result = self.ocr_processor.extract_text(pdf_path)
            
            # 2. Field Extraction
            self.logger.info("Extracting fields...")
            extracted_fields = self.field_extractor.extract_fields_and_labels(ocr_result)
            
            # 3. Table Detection
            self.logger.info("Detecting tables...")
            table_data = self.table_detector.detect_tables(ocr_result)
            
            # 4. Feature Generation with local model path
            self.logger.info("Generating features...")
            features = self.feature_generator.generate_features(
                ocr_result=ocr_result,
                key_value_data=extracted_fields,
                table_data=table_data
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Get feature statistics safely
            feature_stats = {
                'num_tokens': features['input_ids'].size(1) if isinstance(features.get('input_ids'), torch.Tensor) else 0,
                'num_words': len(features.get('words', [])),
                'num_fields': len(extracted_fields),
                'num_tables': len(table_data.get('tables', [])),
                'processing_time': processing_time
            }
            
            # Create serializable features dict
            serializable_features = {
                'input_ids': features['input_ids'].tolist() if isinstance(features.get('input_ids'), torch.Tensor) else [],
                'attention_mask': features['attention_mask'].tolist() if isinstance(features.get('attention_mask'), torch.Tensor) else [],
                'token_type_ids': features['token_type_ids'].tolist() if isinstance(features.get('token_type_ids'), torch.Tensor) else [],
                'bbox': features['bbox'].tolist() if isinstance(features.get('bbox'), torch.Tensor) else [],
                'words': features.get('words', []),
                'boxes': features.get('boxes', [])
            }
            
            # Prepare result
            result = {
                'status': 'success',
                'processing_time': processing_time,
                'output_path': os.path.join(self.output_dir, 'dataset'),
                'statistics': feature_stats,
                'extracted_fields': extracted_fields,
                'table_data': table_data,
                'features': serializable_features,
                'model_path': self.model_path  # Include model path in result
            }
            
            # Validate output
            if not self.validate_output(result):
                raise ValueError("Invalid output format")
            
            # Log results
            self._log_processing_results(pdf_path, result)
            
            self.logger.info(f"Document processing completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            self.logger.error(error_msg)
            return {
                'status': 'error',
                'message': error_msg
            }

    def _log_processing_results(self, pdf_path: str, result: Dict):
        """
        Log document processing results
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.basename(pdf_path)
            log_filename = f'doc_processing_log_{filename}_{timestamp}.json'
            log_path = os.path.join(self.doc_log_dir, log_filename)
            
            # Create serializable log data
            log_data = {
                'timestamp': timestamp,
                'document_name': filename,
                'processing_time': result['processing_time'],
                'status': result['status'],
                'statistics': result['statistics'],
                'output_path': result['output_path'],
                'model_path': self.model_path,
                'extracted_fields': result['extracted_fields'],
                'table_data': {
                    'num_tables': len(result['table_data'].get('tables', [])),
                    'tables': result['table_data'].get('tables', [])
                }
            }
            
            # Save log file
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2)
            
            self.logger.info(f"Processing results logged to {log_path}")
            
        except Exception as e:
            self.logger.error(f"Error logging processing results: {str(e)}")

    def validate_output(self, result: Dict) -> bool:
        """
        Validate processing output
        """
        try:
            # Check basic structure
            if result.get('status') != 'success':
                return False
                
            required_fields = [
                'processing_time',
                'output_path',
                'statistics',
                'features',
                'extracted_fields',
                'table_data',
                'model_path'  # Added model_path validation
            ]
            
            for field in required_fields:
                if field not in result:
                    self.logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate features structure
            features = result['features']
            required_feature_fields = [
                'input_ids',
                'attention_mask',
                'token_type_ids',
                'bbox',
                'words',
                'boxes'
            ]
            
            for field in required_feature_fields:
                if field not in features:
                    self.logger.error(f"Missing required feature field: {field}")
                    return False
            
            # Validate statistics
            required_stats = [
                'num_tokens',
                'num_words',
                'num_fields',
                'num_tables',
                'processing_time'
            ]
            
            for stat in required_stats:
                if stat not in result['statistics']:
                    self.logger.error(f"Missing required statistic: {stat}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating output: {str(e)}")
            return False

    def batch_process_documents(self, pdf_dir: str) -> Dict:
        """
        Process multiple documents in a directory
        """
        try:
            results = []
            failed_docs = []
            start_time = datetime.now()
            
            # Process all PDFs in directory
            for filename in os.listdir(pdf_dir):
                if filename.lower().endswith('.pdf'):
                    pdf_path = os.path.join(pdf_dir, filename)
                    self.logger.info(f"Processing document: {filename}")
                    
                    result = self.process_document(pdf_path)
                    
                    if result['status'] == 'success':
                        results.append({
                            'filename': filename,
                            'result': result
                        })
                    else:
                        failed_docs.append({
                            'filename': filename,
                            'error': result['message']
                        })
            
            # Calculate total processing time
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare batch processing summary
            summary = {
                'status': 'completed',
                'total_documents': len(results) + len(failed_docs),
                'successful': len(results),
                'failed': len(failed_docs),
                'total_processing_time': total_time,
                'average_processing_time': total_time / (len(results) + len(failed_docs)) if results or failed_docs else 0,
                'failed_documents': failed_docs,
                'output_directory': self.output_dir,
                'model_path': self.model_path
            }
            
            return summary
            
        except Exception as e:
            error_msg = f"Error in batch processing: {str(e)}"
            self.logger.error(error_msg)
            return {
                'status': 'error',
                'message': error_msg
            }

