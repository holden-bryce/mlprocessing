import os
import logging
from datetime import datetime
import json
import pytest
from typing import Dict

# Import processors
from src.processors.document_processor import DocumentProcessor
from src.processors.ocr_processor import OCRProcessor
from src.processors.field_extractor import FieldExtractor
from src.processors.table_detector import TableDetector
from src.processors.feature_generator import FeatureGenerator
from src.processors.dataset_preparer import DatasetPreparer

class TestDocumentProcessor:
    @classmethod
    def setup_class(cls):
        """Set up test environment and paths"""
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger(__name__)
        
        # Set up paths
        cls.current_dir = os.path.dirname(os.path.abspath(__file__))
        cls.project_root = os.path.dirname(cls.current_dir)
        
        # Define all test paths
        cls.test_paths = {
            'pdf': os.path.join(cls.project_root, 'data', 'raw', 'docu-1.pdf'),  # Updated filename
            'output': os.path.join(cls.project_root, 'data', 'processed'),
            'models': os.path.join(cls.project_root, 'models', 'layoutlm-base-uncased'),
            'logs': os.path.join(cls.project_root, 'logs', 'test_logs')
        }
        
        # Create necessary directories
        for path in cls.test_paths.values():
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Validate test PDF exists
        if not os.path.exists(cls.test_paths['pdf']):
            raise FileNotFoundError(f"Test PDF not found at {cls.test_paths['pdf']}")

    def setup_method(self):
        """Initialize test components"""
        self.logger.info("Initializing test components...")
        
        try:
            # Initialize processors with local model path
            self.ocr_processor = OCRProcessor(
                output_dir=self.test_paths['output']
            )
            
            self.field_extractor = FieldExtractor(
                output_dir=self.test_paths['output']
            )
            
            self.table_detector = TableDetector(
                output_dir=self.test_paths['output']
            )
            
            self.feature_generator = FeatureGenerator(
                output_dir=self.test_paths['output']
            )
            
            self.dataset_preparer = DatasetPreparer(
                output_dir=self.test_paths['output']
            )
            
            # Initialize document processor
            self.doc_processor = DocumentProcessor(
                ocr_processor=self.ocr_processor,
                field_extractor=self.field_extractor,
                table_detector=self.table_detector,
                feature_generator=self.feature_generator,
                output_dir=self.test_paths['output']
            )
            
            self.logger.info("Test components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing test components: {str(e)}")
            raise

    def test_ocr_processing(self) -> Dict:
        """Test OCR processing"""
        self.logger.info("Testing OCR processing...")
        try:
            ocr_result = self.ocr_processor.extract_text(self.test_paths['pdf'])
            assert ocr_result is not None
            assert 'Blocks' in ocr_result
            self.logger.info("OCR processing test passed ✓")
            return ocr_result
        except Exception as e:
            self.logger.error(f"OCR processing test failed: {str(e)}")
            raise

    def test_field_extraction(self, ocr_result: Dict) -> Dict:
        """Test field extraction"""
        self.logger.info("Testing field extraction...")
        try:
            fields = self.field_extractor.extract_fields_and_labels(ocr_result)
            assert fields is not None
            assert isinstance(fields, dict)
            self.logger.info("Field extraction test passed ✓")
            return fields
        except Exception as e:
            self.logger.error(f"Field extraction test failed: {str(e)}")
            raise

    def test_table_detection(self, ocr_result: Dict) -> Dict:
        """Test table detection"""
        self.logger.info("Testing table detection...")
        try:
            tables = self.table_detector.detect_tables(ocr_result)
            assert tables is not None
            assert isinstance(tables, dict)
            assert 'tables' in tables
            self.logger.info("Table detection test passed ✓")
            return tables
        except Exception as e:
            self.logger.error(f"Table detection test failed: {str(e)}")
            raise

    def test_feature_generation(self, ocr_result: Dict, fields: Dict, tables: Dict) -> Dict:
        """Test feature generation"""
        self.logger.info("Testing feature generation...")
        try:
            features = self.feature_generator.generate_features(
                ocr_result=ocr_result,
                key_value_data=fields,
                table_data=tables
            )
            assert features is not None
            assert 'input_ids' in features
            assert 'attention_mask' in features
            assert 'bbox' in features
            self.logger.info("Feature generation test passed ✓")
            return features
        except Exception as e:
            self.logger.error(f"Feature generation test failed: {str(e)}")
            raise

    def test_dataset_preparation(self, features: Dict) -> Dict:
        """Test dataset preparation"""
        self.logger.info("Testing dataset preparation...")
        try:
            training_features = self.dataset_preparer.prepare_training_data(features)
            assert training_features is not None
            assert 'input_ids' in training_features
            assert 'attention_mask' in training_features
            assert 'bbox' in training_features
            self.logger.info("Dataset preparation test passed ✓")
            return training_features
        except Exception as e:
            self.logger.error(f"Dataset preparation test failed: {str(e)}")
            raise

    def test_full_pipeline(self):
        """Test the complete document processing pipeline"""
        self.logger.info("Testing full document processing pipeline...")
        try:
            # Process document
            result = self.doc_processor.process_document(self.test_paths['pdf'])
            
            # Validate result
            assert result['status'] == 'success'
            assert 'processing_time' in result
            assert 'features' in result
            assert 'extracted_fields' in result
            assert 'table_data' in result
            assert 'model_path' in result
            
            # Log test results
            self._log_test_results(result)
            
            self.logger.info("Full pipeline test passed ✓")
            return result
        except Exception as e:
            self.logger.error(f"Full pipeline test failed: {str(e)}")
            raise

    def _log_test_results(self, result: Dict):
        """Log test results"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(
                self.test_paths['logs'],
                f'test_results_{timestamp}.json'
            )
            
            test_summary = {
                'timestamp': timestamp,
                'pdf_path': self.test_paths['pdf'],
                'model_path': self.test_paths['models'],
                'processing_time': result['processing_time'],
                'statistics': result['statistics'],
                'status': 'passed'
            }
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(test_summary, f, indent=2)
            
            # Print summary
            print("\n" + "="*50)
            print("TEST SUMMARY")
            print("="*50)
            print(f"Processing Time: {result['processing_time']:.2f} seconds")
            print(f"Fields Extracted: {result['statistics']['num_fields']}")
            print(f"Tables Detected: {result['statistics']['num_tables']}")
            print(f"Tokens Generated: {result['statistics']['num_tokens']}")
            print(f"Words Processed: {result['statistics']['num_words']}")
            print(f"Model Path: {self.test_paths['models']}")
            print("="*50 + "\n")
            
        except Exception as e:
            self.logger.error(f"Error logging test results: {str(e)}")

def main():
    """Main test function"""
    test = TestDocumentProcessor()
    test.setup_class()
    test.setup_method()
    
    try:
        # Run individual component tests
        ocr_result = test.test_ocr_processing()
        fields = test.test_field_extraction(ocr_result)
        tables = test.test_table_detection(ocr_result)
        features = test.test_feature_generation(ocr_result, fields, tables)
        training_features = test.test_dataset_preparation(features)
        
        # Run full pipeline test
        result = test.test_full_pipeline()
        
        print("\nAll tests completed successfully! 🎉")
        return True
        
    except Exception as e:
        print(f"\nTests failed: {str(e)} ❌")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 