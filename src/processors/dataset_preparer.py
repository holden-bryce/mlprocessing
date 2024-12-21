import logging
import os
from typing import Dict, List
import torch
from transformers import LayoutLMTokenizer
import json
from datetime import datetime
import pandas as pd

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
        
        # Initialize dataset collection
        self.dataset_file = os.path.join(self.dataset_dir, 'complete_dataset.pt')
        self.dataset_summary = os.path.join(self.dataset_dir, 'dataset_summary.json')
        self.collected_data = []

    def prepare_training_data(self, features: Dict, document_info: Dict = None) -> Dict:
        """Prepare features for training and add to dataset"""
        try:
            # Process features
            training_features = self._process_features(features)
            
            # Add document info if provided
            if document_info:
                training_features['document_info'] = document_info
            
            # Add to collection
            self.collected_data.append(training_features)
            
            # Save updated dataset
            self._save_complete_dataset()
            
            return training_features

        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            raise

    def _process_features(self, features: Dict) -> Dict:
        """Process and standardize features"""
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

        return {
            'input_ids': input_ids,
            'attention_mask': features['attention_mask'],
            'token_type_ids': features['token_type_ids'],
            'bbox': features['bbox'],
            'labels': features.get('labels', None),
            'words': features.get('words', []),
            'boxes': features.get('boxes', []),
            'timestamp': datetime.now().isoformat()
        }

    def _save_complete_dataset(self):
        """Save complete dataset and summary"""
        try:
            # Save tensor dataset
            torch.save({
                'data': self.collected_data,
                'model_path': self.model_path,
                'last_updated': datetime.now().isoformat()
            }, self.dataset_file)
            
            # Create and save summary
            summary = {
                'num_documents': len(self.collected_data),
                'last_updated': datetime.now().isoformat(),
                'model_path': self.model_path,
                'max_seq_length': self.max_seq_length,
                'features_per_document': [
                    {
                        'timestamp': data['timestamp'],
                        'num_words': len(data['words']),
                        'num_boxes': len(data['boxes'])
                    }
                    for data in self.collected_data
                ]
            }
            
            with open(self.dataset_summary, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Dataset saved with {len(self.collected_data)} documents")
            
        except Exception as e:
            self.logger.error(f"Error saving complete dataset: {str(e)}")
            raise

    def load_complete_dataset(self) -> Dict:
        """Load complete dataset"""
        try:
            if os.path.exists(self.dataset_file):
                dataset = torch.load(self.dataset_file)
                self.logger.info(f"Loaded dataset with {len(dataset['data'])} documents")
                return dataset
            else:
                self.logger.warning("No dataset file found")
                return None
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise 