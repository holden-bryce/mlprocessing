import logging
import os
from typing import Dict, List
import torch
from transformers import LayoutLMTokenizer
import json
from datetime import datetime

class DatasetPreparer:
    def __init__(self, 
                 output_dir: str,
                 max_seq_length: int = 512):
        """Initialize DatasetPreparer"""
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
        
        # Create dataset directory
        self.dataset_dir = os.path.join(output_dir, 'dataset')
        os.makedirs(self.dataset_dir, exist_ok=True)

    def prepare_training_data(self, features: Dict) -> Dict:
        """Prepare features for training"""
        try:
            # Get special token ids
            cls_token_id = self.tokenizer.cls_token_id
            sep_token_id = self.tokenizer.sep_token_id
            pad_token_id = self.tokenizer.pad_token_id

            # Process input ids
            input_ids = features['input_ids']
            if input_ids[0][0] != cls_token_id:
                input_ids = torch.cat([
                    torch.tensor([[cls_token_id]]),
                    input_ids,
                    torch.tensor([[sep_token_id]])
                ], dim=1)

            # Pad if needed
            if input_ids.size(1) < self.max_seq_length:
                padding_length = self.max_seq_length - input_ids.size(1)
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[pad_token_id] * padding_length])
                ], dim=1)

            # Prepare training features
            training_features = {
                'input_ids': input_ids,
                'attention_mask': features['attention_mask'],
                'token_type_ids': features['token_type_ids'],
                'bbox': features['bbox'],
                'labels': features.get('labels', None)
            }

            # Save features
            self._save_features(training_features)

            return training_features

        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            raise

    def _save_features(self, features: Dict):
        """Save features to disk"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'training_features_{timestamp}.pt'
            output_path = os.path.join(self.dataset_dir, filename)

            # Save tensor data
            torch.save({
                'input_ids': features['input_ids'],
                'attention_mask': features['attention_mask'],
                'token_type_ids': features['token_type_ids'],
                'bbox': features['bbox'],
                'labels': features['labels'],
                'model_path': self.model_path
            }, output_path)

            self.logger.info(f"Training features saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving features: {str(e)}")
            raise

    def load_training_data(self, feature_file: str) -> Dict:
        """Load training features from disk"""
        try:
            feature_path = os.path.join(self.dataset_dir, feature_file)
            features = torch.load(feature_path)
            
            self.logger.info(f"Loaded training features from {feature_path}")
            return features

        except Exception as e:
            self.logger.error(f"Error loading training data: {str(e)}")
            raise 