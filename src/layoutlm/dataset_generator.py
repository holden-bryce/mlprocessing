from typing import Dict, List, Optional, Union
from pathlib import Path
import json
import logging
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from tqdm import tqdm
import os
from datetime import datetime

@dataclass
class LayoutLMExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    bbox: torch.Tensor
    labels: Optional[torch.Tensor] = None

class LayoutLMDataset(Dataset):
    def __init__(self, examples: List[LayoutLMExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        return {
            "input_ids": example.input_ids,
            "attention_mask": example.attention_mask,
            "token_type_ids": example.token_type_ids,
            "bbox": example.bbox,
            "labels": example.labels
        }

class DatasetGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate(self, results: List[Dict], output_dir: Union[str, Path]) -> None:
        """Generate dataset from processing results.
        
        Args:
            results: List of document processing results
            output_dir: Output directory for dataset files
        """
        try:
            self.logger.info("Generating dataset...")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Convert results to examples
            examples = []
            for result in tqdm(results, desc="Converting results to examples"):
                if not result.get('layoutlm_features'):
                    self.logger.warning(f"Skipping result without LayoutLM features")
                    continue

                features = result['layoutlm_features']
                example = LayoutLMExample(
                    input_ids=torch.tensor(features['input_ids']),
                    attention_mask=torch.tensor(features['attention_mask']),
                    token_type_ids=torch.tensor(features['token_type_ids']),
                    bbox=torch.tensor(features['bbox']),
                    labels=torch.tensor(features['labels']) if features.get('labels') else None
                )
                examples.append(example)

            # Create dataset
            dataset = LayoutLMDataset(examples)

            # Save dataset
            output_path = output_dir / "dataset.pt"
            torch.save(dataset, output_path)
            self.logger.info(f"Dataset saved to {output_path}")

            # Save metadata
            metadata = {
                "num_examples": len(examples),
                "features": list(examples[0].__dict__.keys()) if examples else [],
                "creation_date": datetime.now().isoformat()
            }
            
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Generated dataset with {len(examples)} examples")

        except Exception as e:
            self.logger.error(f"Failed to generate dataset: {str(e)}")
            raise 