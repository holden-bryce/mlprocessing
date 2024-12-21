import torch
import json
import os
from datetime import datetime
from pprint import pprint

def load_and_view_dataset(dataset_path='data/processed/dataset/layoutlm_training_dataset.pt', save_json=True):
    """Load and view the processed dataset"""
    try:
        # Load the PyTorch dataset
        print(f"\nLoading dataset from: {dataset_path}")
        dataset = torch.load(dataset_path)
        
        # Print basic info
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        print(f"Number of Documents: {len(dataset['data'])}")
        print(f"Last Updated: {dataset['last_updated']}")
        print(f"Model Path: {dataset['model_path']}")
        
        # Create readable version of the data
        readable_data = []
        for idx, doc in enumerate(dataset['data']):
            readable_doc = {
                'document_id': idx + 1,
                'timestamp': doc['timestamp'],
                'document_info': doc.get('document_info', {}),
                'statistics': {
                    'num_words': len(doc['words']),
                    'num_boxes': len(doc['boxes']),
                },
                'extracted_fields': doc['document_info'].get('extracted_fields', {}),
                'table_data': doc['document_info'].get('table_data', {}),
                'text_content': ' '.join(doc['words'])[:200] + '...' # First 200 chars
            }
            readable_data.append(readable_doc)
        
        # Print detailed info for first document
        print("\n" + "="*50)
        print("FIRST DOCUMENT DETAILS")
        print("="*50)
        pprint(readable_data[0], indent=2)
        
        # Save as JSON if requested
        if save_json:
            json_path = os.path.join(os.path.dirname(dataset_path), 'dataset_readable.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'dataset_info': {
                        'num_documents': len(dataset['data']),
                        'last_updated': dataset['last_updated'],
                        'model_path': dataset['model_path']
                    },
                    'documents': readable_data
                }, f, indent=2)
            print(f"\nReadable version saved to: {json_path}")
        
        return dataset, readable_data
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None, None

def view_document_details(doc_idx, dataset_path='data/processed/dataset/complete_dataset.pt'):
    """View details for a specific document"""
    dataset = torch.load(dataset_path)
    if 0 <= doc_idx < len(dataset['data']):
        doc = dataset['data'][doc_idx]
        print("\n" + "="*50)
        print(f"DOCUMENT {doc_idx + 1} DETAILS")
        print("="*50)
        
        # Document info
        print("\nDocument Info:")
        pprint(doc['document_info'])
        
        # Text content
        print("\nText Content (first 200 chars):")
        print(' '.join(doc['words'])[:200] + '...')
        
        # Fields
        print("\nExtracted Fields:")
        pprint(doc['document_info'].get('extracted_fields', {}))
        
        # Tables
        print("\nTable Data:")
        pprint(doc['document_info'].get('table_data', {}))
        
        # Tensor shapes
        print("\nTensor Shapes:")
        for key, value in doc.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
    else:
        print(f"Error: Document index {doc_idx} out of range")

if __name__ == "__main__":
    # Load and view dataset summary
    dataset, readable_data = load_and_view_dataset()
    
    # View specific document (e.g., first document)
    print("\nViewing first document details:")
    view_document_details(0)