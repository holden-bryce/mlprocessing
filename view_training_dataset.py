import torch
import json
from pprint import pprint

def view_training_dataset():
    # Load the training dataset
    dataset = torch.load('data/processed/dataset/layoutlm_training_dataset.pt')
    
    # Load the summary
    with open('data/processed/dataset/training_dataset_summary.json', 'r') as f:
        summary = json.load(f)
    
    print("\n=== DATASET SUMMARY ===")
    print(f"Number of examples: {summary['num_examples']}")
    print("\nFeature dimensions:")
    pprint(summary['feature_dimensions'])
    
    print("\n=== FIRST EXAMPLE DETAILS ===")
    example = dataset['examples'][0]
    for key, value in example.items():
        print(f"\n{key}:")
        print(f"Shape: {value.shape}")
        print(f"Type: {value.dtype}")
        if key == 'input_ids':
            print("First 50 tokens:", value[:50].tolist())
    
    print("\n=== LABEL MAP ===")
    pprint(dataset['label_map'])

if __name__ == "__main__":
    view_training_dataset() 