from typing import Dict, List, Optional
import json
import re
from pathlib import Path
import logging

class FieldExtractor:
    def __init__(self, config_path: str = "config/field_patterns.json"):
        """Initialize field extractor with patterns from config."""
        self.logger = logging.getLogger(__name__)
        
        # Load field patterns from config
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.field_patterns = config.get('fields', {})

    def extract_fields(self, ocr_data: Dict) -> Dict:
        """Extract fields from OCR data using patterns.
        
        Args:
            ocr_data: Dictionary containing OCR results with 'text', 'boxes', and 'confidences'
            
        Returns:
            Dictionary of extracted fields with their values and metadata
        """
        extracted_fields = {}
        
        try:
            texts = ocr_data.get('text', [])
            boxes = ocr_data.get('boxes', [])
            confidences = ocr_data.get('confidences', [])
            
            if not texts or not boxes or not confidences:
                self.logger.warning("Missing OCR data components")
                return extracted_fields

            # Process each field pattern
            for field_name, field_config in self.field_patterns.items():
                field_type = field_config.get('type', 'string')
                required = field_config.get('required', False)
                
                # Find field in OCR text
                field_value = None
                field_confidence = 0.0
                field_box = None
                
                # Iterate through OCR results
                for i, (text, box, conf) in enumerate(zip(texts, boxes, confidences)):
                    if not text.strip():  # Skip empty text
                        continue
                        
                    # Match field pattern
                    if self._match_field_pattern(text, field_name, field_type):
                        field_value = text
                        field_confidence = conf
                        field_box = box
                        break
                
                # Add extracted field
                if field_value or required:
                    extracted_fields[field_name] = {
                        'value': field_value,
                        'confidence': field_confidence,
                        'bbox': field_box,
                        'type': field_type
                    }
            
            return extracted_fields
            
        except Exception as e:
            self.logger.error(f"Field extraction failed: {str(e)}")
            raise

    def _match_field_pattern(self, text: str, field_name: str, field_type: str) -> bool:
        """Match text against field pattern based on type."""
        try:
            if field_type == 'date':
                # Match date patterns
                date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
                return bool(re.search(date_pattern, text))
                
            elif field_type == 'currency':
                # Match currency patterns
                currency_pattern = r'\$?\d+(?:\.\d{2})?'
                return bool(re.search(currency_pattern, text))
                
            elif field_type == 'multi-line':
                # Match address-like patterns
                return len(text.split()) > 3
                
            else:  # string type
                # Simple string matching
                return field_name.lower() in text.lower()
                
        except Exception as e:
            self.logger.error(f"Pattern matching failed: {str(e)}")
            return False 