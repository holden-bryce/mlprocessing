import logging
from typing import Dict, List, Tuple
import torch
from transformers import LayoutLMTokenizer
import os
import json
from datetime import datetime

class FeatureGenerator:
    def __init__(self, 
                 output_dir: str,
                 max_seq_length: int = 512):
        """Initialize FeatureGenerator"""
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.logger = logging.getLogger(__name__)
        
        # Set up model path
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = os.path.join(self.project_root, 'models', 'layoutlm-base-uncased')
        
        # Initialize tokenizer
        try:
            self.tokenizer = LayoutLMTokenizer.from_pretrained(self.model_path)
        except Exception as e:
            self.logger.warning(f"Could not load tokenizer from {self.model_path}, using default")
            self.tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
            # Save it locally for next time
            os.makedirs(self.model_path, exist_ok=True)
            self.tokenizer.save_pretrained(self.model_path)
        
        # Create feature logs directory
        self.feature_log_dir = os.path.join(output_dir, 'feature_logs')
        os.makedirs(self.feature_log_dir, exist_ok=True)

    def generate_features(self, ocr_result: Dict, key_value_data: Dict = None, table_data: Dict = None) -> Dict:
        """Generate LayoutLM features from OCR results"""
        try:
            # Extract blocks from OCR result
            blocks = ocr_result.get('Blocks', [])
            
            # Process text and layout information
            words, boxes, normalized_boxes = self._process_blocks(blocks)
            
            # Add key-value and table information to words if available
            if key_value_data:
                words = self._add_field_markers(words, key_value_data)
            
            if table_data:
                words = self._add_table_markers(words, table_data)
            
            # Generate features
            features = self._convert_to_features(
                words=words,
                boxes=boxes,
                normalized_boxes=normalized_boxes
            )
            
            # Log feature generation
            self._log_features(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error generating features: {str(e)}")
            raise

    def _process_blocks(self, blocks: List[Dict]) -> Tuple[List[str], List[List[int]], List[List[int]]]:
        """Process Textract blocks into words and bounding boxes"""
        words = []
        boxes = []
        normalized_boxes = []
        
        for block in blocks:
            if block['BlockType'] == 'WORD':
                # Get text
                words.append(block['Text'])
                
                # Get bounding box
                if 'Geometry' in block and 'BoundingBox' in block['Geometry']:
                    bbox = block['Geometry']['BoundingBox']
                    
                    # Original coordinates
                    box = [
                        int(bbox['Left'] * 1000),
                        int(bbox['Top'] * 1000),
                        int((bbox['Left'] + bbox['Width']) * 1000),
                        int((bbox['Top'] + bbox['Height']) * 1000)
                    ]
                    boxes.append(box)
                    
                    # Normalized coordinates
                    normalized_boxes.append(box)  # Already normalized by Textract
                else:
                    boxes.append([0, 0, 0, 0])
                    normalized_boxes.append([0, 0, 0, 0])
        
        return words, boxes, normalized_boxes

    def _add_field_markers(self, words: List[str], key_value_data: Dict) -> List[str]:
        """Add markers for key-value pairs"""
        marked_words = words.copy()
        for field_name, field_value in key_value_data.items():
            if isinstance(field_value, str):
                for i, word in enumerate(marked_words):
                    if word in field_value:
                        marked_words[i] = f"[FIELD_{field_name}]{word}"
        return marked_words

    def _add_table_markers(self, words: List[str], table_data: Dict) -> List[str]:
        """Add markers for table cells"""
        marked_words = words.copy()
        for table in table_data.get('tables', []):
            for cell in table.get('cells', []):
                cell_text = cell.get('text', '')
                for i, word in enumerate(marked_words):
                    if word in cell_text:
                        cell_type = 'HEADER' if cell.get('is_header', False) else 'CELL'
                        marked_words[i] = f"[TABLE_{cell_type}]{word}"
        return marked_words

    def _convert_to_features(self, 
                           words: List[str],
                           boxes: List[List[int]],
                           normalized_boxes: List[List[int]]) -> Dict:
        """Convert processed text and layout information to LayoutLM features"""
        # Join words with spaces
        text = ' '.join(words)
        
        # Tokenize text
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        
        # Get token boxes
        token_boxes = []
        for word_idx, word in enumerate(words):
            word_tokens = self.tokenizer.tokenize(word)
            token_boxes.extend([normalized_boxes[word_idx]] * len(word_tokens))
        
        # Add special tokens boxes
        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[0, 0, 0, 0]]
        
        # Pad boxes if necessary
        while len(token_boxes) < self.max_seq_length:
            token_boxes.append([0, 0, 0, 0])
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'token_type_ids': encoded['token_type_ids'],
            'bbox': torch.tensor([token_boxes]),
            'words': words,
            'boxes': boxes
        }

    def _log_features(self, features: Dict):
        """Log generated features"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'feature_log_{timestamp}.json'
            log_path = os.path.join(self.feature_log_dir, log_filename)
            
            # Convert tensors to lists for JSON serialization
            log_data = {
                'timestamp': timestamp,
                'model_path': self.model_path,
                'feature_stats': {
                    'num_tokens': features['input_ids'].shape[1],
                    'num_words': len(features['words']),
                    'num_boxes': len(features['boxes'])
                }
            }
            
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2)
            
            self.logger.info(f"Features logged to {log_path}")
            
        except Exception as e:
            self.logger.error(f"Error logging features: {str(e)}") 