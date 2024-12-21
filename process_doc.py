import os
import logging
from src.processors.document_processor import DocumentProcessor
from src.processors.ocr_processor import OCRProcessor
from src.processors.field_extractor import FieldExtractor
from src.processors.table_detector import TableDetector
from src.processors.feature_generator import FeatureGenerator

def process_document(pdf_path: str):
    # Set up output directory
    output_dir = os.path.join('data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    ocr_processor = OCRProcessor(output_dir=output_dir)
    field_extractor = FieldExtractor(output_dir=output_dir)
    table_detector = TableDetector(output_dir=output_dir)
    feature_generator = FeatureGenerator(output_dir=output_dir)
    
    # Initialize document processor
    doc_processor = DocumentProcessor(
        ocr_processor=ocr_processor,
        field_extractor=field_extractor,
        table_detector=table_detector,
        feature_generator=feature_generator,
        output_dir=output_dir
    )
    
    # Process document
    result = doc_processor.process_document(pdf_path)
    
    # Print results
    if result['status'] == 'success':
        print("\nProcessing completed successfully! 🎉")
        print(f"Processing Time: {result['processing_time']:.2f} seconds")
        print(f"Fields Extracted: {result['statistics']['num_fields']}")
        print(f"Tables Detected: {result['statistics']['num_tables']}")
    else:
        print(f"\nProcessing failed: {result.get('message', 'Unknown error')} ❌")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python process_doc.py path/to/document.pdf")
    else:
        process_document(sys.argv[1]) 