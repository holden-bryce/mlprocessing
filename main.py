import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime

from src.processors.document_processor import DocumentProcessor
from src.layoutlm.dataset_generator import DatasetGenerator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DocumentProcessingPipeline:
    def __init__(self, config_path: str):
        """Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        
        # Load config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.processor = DocumentProcessor(config_path)
        self.dataset_generator = DatasetGenerator()
        
    def process_documents(self, 
                         input_dir: str,
                         output_dir: str,
                         file_pattern: str = "*.pdf") -> List[Dict]:
        """Process all documents in input directory.
        
        Args:
            input_dir: Directory containing input documents
            output_dir: Directory for output files
            file_pattern: Pattern to match input files
            
        Returns:
            List of processing results
        """
        try:
            input_path = Path(input_dir)
            if not input_path.exists():
                raise FileNotFoundError(f"Input directory not found: {input_dir}")
                
            # Get list of files to process
            files = list(input_path.glob(file_pattern))
            if not files:
                raise ValueError(f"No files found matching pattern: {file_pattern}")
                
            logger.info(f"Found {len(files)} files to process")
            
            # Process each file
            results = []
            for file_path in files:
                try:
                    result = self.process_single_document(file_path, output_dir)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    continue
                    
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

    def process_single_document(self, 
                              file_path: str,
                              output_dir: Optional[str] = None) -> Dict:
        """Process a single document.
        
        Args:
            file_path: Path to document file
            output_dir: Optional output directory for results
            
        Returns:
            Processing results
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Document not found: {file_path}")
            
            logger.info(f"Processing document: {file_path}")
            result = self.processor.process_document(file_path, output_dir)
            self._validate_result(result)
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise

    def generate_dataset(self, 
                        results: List[Dict],
                        dataset_dir: str) -> None:
        """Generate dataset from processing results.
        
        Args:
            results: List of document processing results
            dataset_dir: Output directory for dataset
        """
        try:
            # Create output directory
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Generate dataset
            self.dataset_generator.generate(results, dataset_dir)
            
            # Save dataset metadata
            metadata = {
                "creation_date": datetime.now().isoformat(),
                "num_documents": len(results),
                "config_path": self.config_path
            }
            
            metadata_path = Path(dataset_dir) / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Dataset generated in {dataset_dir}")
            
        except Exception as e:
            logger.error(f"Dataset generation failed: {str(e)}")
            raise

    def _validate_result(self, result: Dict) -> None:
        """Validate processing results against config rules.
        
        Args:
            result: Processing results to validate
        """
        validation_rules = self.config.get('validation', {}).get('rules', {})
        confidence_thresholds = self.config.get('validation', {}).get('confidence_thresholds', {})
        
        # Basic validation of required fields
        if not result.get('ocr_results'):
            raise ValueError("Missing OCR results")
        
        if not result.get('extracted_fields'):
            raise ValueError("No fields were extracted")
        
        # Validate required fields from config
        required_fields = [
            field_name for field_name, field_config in self.config.get('fields', {}).items()
            if field_config.get('required', False)
        ]
        
        missing_fields = [
            field for field in required_fields
            if field not in result.get('extracted_fields', {})
        ]
        
        if missing_fields:
            error_handling = self.config.get('logging', {}).get('error_handling', {})
            if error_handling.get('missing_fields') == 'error':
                raise ValueError(f"Missing required fields: {missing_fields}")
            else:
                logger.warning(f"Missing required fields: {missing_fields}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Document Processing Pipeline")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input documents"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for processed results"
    )
    
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Directory for generated dataset"
    )
    
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*.pdf",
        help="Pattern to match input files"
    )
    
    parser.add_argument(
        "--local-model-path",
        type=str,
        help="Path to local model directory"
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Initialize pipeline
        pipeline = DocumentProcessingPipeline(args.config)
        
        # Process documents
        results = pipeline.process_documents(
            args.input_dir,
            args.output_dir,
            args.file_pattern
        )
        
        # Generate dataset
        pipeline.generate_dataset(results, args.dataset_dir)
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 