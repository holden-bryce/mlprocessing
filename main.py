import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime
import sys

from src.processors.document_processor import DocumentProcessor
from src.layoutlm.dataset_generator import DatasetGenerator
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger(__name__)

class Pipeline:
    def __init__(self, config_path: str):
        """Initialize the document processing pipeline"""
        self.config = self._load_config(config_path)
        self.processor = DocumentProcessor(
            config_path=config_path,
            model_path=self.config.get('model_settings', {}).get(
                'model_path', 
                'microsoft/layoutlm-base-uncased'
            )
        )
        self.dataset_generator = DatasetGenerator(
            output_dir=self.config.get('paths', {}).get('dataset_dir', 'dataset')
        )

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    def process_documents(self, input_dir: Path, output_dir: Path) -> List[Dict]:
        """Process all documents in the input directory"""
        processed_docs = []
        pdf_files = list(input_dir.glob('*.pdf'))
        
        total_files = len(pdf_files)
        logger.info(f"Found {total_files} PDF files to process")
        
        for index, pdf_path in enumerate(pdf_files, 1):
            try:
                logger.info(f"Processing document {index}/{total_files}: {pdf_path}")
                
                # Generate output filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = output_dir / f"processed_{pdf_path.stem}_{timestamp}.json"
                
                # Process document
                result = self.processor.process_document(
                    str(pdf_path),
                    str(output_file)
                )
                
                # Add metadata
                result['metadata'] = {
                    'source_file': str(pdf_path),
                    'processing_date': timestamp,
                    'output_file': str(output_file)
                }
                
                processed_docs.append(result)
                
                # Save individual result
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved processing results to {output_file}")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                continue
                
        return processed_docs

    def generate_dataset(self, processed_docs: List[Dict], dataset_dir: Path) -> None:
        """Generate LayoutLM dataset from processed documents"""
        try:
            self.dataset_generator.generate_dataset(processed_docs)
            
            # Save dataset metadata
            metadata = {
                'creation_date': datetime.now().isoformat(),
                'num_documents': len(processed_docs),
                'config_path': str(self.config)
            }
            
            metadata_path = dataset_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Generated dataset with {len(processed_docs)} documents")
            
        except Exception as e:
            logger.error(f"Failed to generate dataset: {e}")
            raise

    def run(self, input_dir: str, output_dir: str, dataset_dir: str) -> None:
        """Run the complete pipeline"""
        try:
            # Create directories
            input_path = Path(input_dir)
            output_path = Path(output_dir)
            dataset_path = Path(dataset_dir)
            
            output_path.mkdir(parents=True, exist_ok=True)
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Process documents
            processed_docs = self.process_documents(input_path, output_path)
            
            if not processed_docs:
                logger.warning("No documents were successfully processed")
                return
            
            # Generate dataset
            self.generate_dataset(processed_docs, dataset_path)
            
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Process documents and generate LayoutLM dataset'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--input-dir', 
        type=str, 
        default='data/raw',
        help='Input directory containing PDF files'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='output',
        help='Output directory for processed results'
    )
    parser.add_argument(
        '--dataset-dir', 
        type=str, 
        default='dataset',
        help='Output directory for LayoutLM dataset'
    )
    return parser.parse_args()

def main():
    """Main entry point"""
    try:
        # Parse arguments
        args = parse_args()
        
        # Initialize and run pipeline
        pipeline = Pipeline(args.config)
        pipeline.run(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            dataset_dir=args.dataset_dir
        )
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 