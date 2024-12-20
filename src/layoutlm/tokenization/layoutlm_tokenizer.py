from transformers import LayoutLMTokenizer, LayoutLMTokenizerFast
from typing import Dict, List, Optional
import torch
import logging
import os

class LayoutLMTokenizer:
    def __init__(self, model_path: str):
        """Initialize LayoutLM tokenizer.
        
        Args:
            model_path: Path to model directory or HuggingFace model name
        """
        self.logger = logging.getLogger(__name__)
        
        try:
            # Check if path is local directory
            if os.path.isdir(model_path):
                # Verify vocab file exists
                vocab_path = os.path.join(model_path, 'vocab.txt')
                if not os.path.exists(vocab_path):
                    raise ValueError(f"vocab.txt not found in {model_path}")
                
                self.logger.info(f"Loading tokenizer from local path: {model_path}")
                try:
                    # Try loading Fast tokenizer first
                    self.tokenizer = LayoutLMTokenizerFast.from_pretrained(model_path)
                except:
                    # Fall back to regular tokenizer
                    self.tokenizer = LayoutLMTokenizer.from_pretrained(model_path)
            else:
                # Assume it's a HuggingFace model name
                self.logger.info(f"Loading tokenizer from HuggingFace: {model_path}")
                try:
                    self.tokenizer = LayoutLMTokenizerFast.from_pretrained(model_path)
                except:
                    self.tokenizer = LayoutLMTokenizer.from_pretrained(model_path)
                    
            self.logger.info("Tokenizer loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {str(e)}")
            raise

    def tokenize(self,
                 words: List[str],
                 boxes: List[List[int]],
                 labels: Optional[List[str]] = None) -> Dict:
        """Tokenize text with layout information."""
        try:
            # Basic validation
            if not words or not boxes:
                raise ValueError("Empty words or boxes provided")
                
            # Convert words to string
            text = ' '.join(str(word) for word in words if word)
            
            # Normalize boxes
            normalized_boxes = [[
                min(max(0, int(coord)), 1000) 
                for coord in box
            ] for box in boxes]
            
            # Encode text
            encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Add bbox information
            encoding['bbox'] = torch.tensor([normalized_boxes[0]] * encoding['input_ids'].size(1))
            
            # Add labels if provided
            if labels:
                label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
                label_ids = [label_map.get(label, 0) for label in labels]
                encoding['labels'] = torch.tensor([label_ids])
            
            return encoding
            
        except Exception as e:
            raise ValueError(f"Tokenization failed: {str(e)}")