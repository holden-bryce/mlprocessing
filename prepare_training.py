import torch
from transformers import LayoutLMTokenizer
import os
from typing import Dict, List
import json

def prepare_layoutlm_dataset(dataset_path='data/processed/dataset/complete_dataset.pt'):
    """Convert processed dataset to LayoutLM training format"""
    
    print("Loading dataset...")
    dataset = torch.load(dataset_path)
    
    # Initialize tokenizer
    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
    
    training_examples = []
    
    for doc in dataset['data']:
        try:
            # Get document info
            doc_info = doc['document_info']
            
            # Fix bbox dimensions - ensure it's [sequence_length, 4]
            bbox = doc['bbox']
            if isinstance(bbox, torch.Tensor):
                if bbox.dim() == 3:
                    bbox = bbox.squeeze(0)  # Remove batch dimension if present
                if bbox.size(1) != 4:
                    bbox = bbox.view(-1, 4)  # Reshape to [sequence_length, 4]
            else:
                # If bbox is a list, convert to tensor with correct shape
                bbox = torch.tensor(bbox).view(-1, 4)
            
            # Prepare input features
            training_example = {
                'input_ids': doc['input_ids'].squeeze(0) if doc['input_ids'].dim() > 1 else doc['input_ids'],
                'attention_mask': doc['attention_mask'].squeeze(0) if doc['attention_mask'].dim() > 1 else doc['attention_mask'],
                'token_type_ids': doc['token_type_ids'].squeeze(0) if doc['token_type_ids'].dim() > 1 else doc['token_type_ids'],
                'bbox': bbox,
                'labels': torch.zeros(doc['input_ids'].size(-1)),  # Match sequence length
            }
            
            # Print shapes for debugging
            print("\nTensor shapes after processing:")
            for key, value in training_example.items():
                print(f"{key}: {value.shape}")
            
            # Verify all tensors have same sequence length
            seq_length = training_example['input_ids'].size(0)
            assert training_example['attention_mask'].size(0) == seq_length
            assert training_example['token_type_ids'].size(0) == seq_length
            assert training_example['bbox'].size(0) == seq_length
            assert training_example['labels'].size(0) == seq_length
            
            training_examples.append(training_example)
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            continue
    
    # Save training-ready dataset
    output_path = os.path.join(os.path.dirname(dataset_path), 'layoutlm_training_dataset.pt')
    torch.save({
        'examples': training_examples,
        'label_map': {
            'O': 0,  # Outside of any field
            'FIELD': 1,  # Generic field
            'TABLE': 2,  # Table content
        },
        'num_labels': 3,
        'model_path': dataset['model_path']
    }, output_path)
    
    print(f"\nTraining dataset saved to: {output_path}")
    print(f"Number of training examples: {len(training_examples)}")
    
    # Save readable summary
    summary = {
        'num_examples': len(training_examples),
        'feature_dimensions': {
            'input_ids': training_examples[0]['input_ids'].shape if training_examples else None,
            'bbox': training_examples[0]['bbox'].shape if training_examples else None,
            'labels': training_examples[0]['labels'].shape if training_examples else None
        },
        'model_path': dataset['model_path']
    }
    
    summary_path = os.path.join(os.path.dirname(dataset_path), 'training_dataset_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return output_path

def validate_training_dataset(dataset_path):
    """Validate that dataset meets LayoutLM requirements"""
    dataset = torch.load(dataset_path)
    
    print("\nValidating training dataset...")
    print("="*50)
    
    # Check required fields
    required_fields = ['input_ids', 'attention_mask', 'token_type_ids', 'bbox', 'labels']
    
    for idx, example in enumerate(dataset['examples']):
        print(f"\nValidating example {idx + 1}:")
        
        for field in required_fields:
            if field not in example:
                print(f"ERROR: Missing required field: {field}")
                return False
            
        # Validate dimensions
        seq_length = example['input_ids'].size(0)
        print(f"Sequence length: {seq_length}")
        
        # Print all tensor shapes
        for field, tensor in example.items():
            print(f"{field} shape: {tensor.shape}")
        
        # All tensors should have same sequence length
        if not all(example[field].size(0) == seq_length for field in required_fields):
            print("ERROR: Mismatched sequence lengths")
            return False
            
        # bbox should be [seq_length, 4]
        if example['bbox'].size() != (seq_length, 4):
            print(f"ERROR: Invalid bbox dimensions: {example['bbox'].size()}")
            return False
    
    print("\nDataset validation successful!")
    print(f"Number of examples: {len(dataset['examples'])}")
    print(f"Number of labels: {dataset['num_labels']}")
    
    return True

if __name__ == "__main__":
    # Convert dataset to training format
    training_path = prepare_layoutlm_dataset()
    
    # Validate training dataset
    validate_training_dataset(training_path) 