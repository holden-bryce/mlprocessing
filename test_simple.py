import os
import logging
from src.processors.document_processor import DocumentProcessor
from src.processors.ocr_processor import OCRProcessor
from src.processors.field_extractor import FieldExtractor
from src.processors.table_detector import TableDetector
from src.processors.feature_generator import FeatureGenerator

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Set up paths
    output_dir = os.path.join('data', 'processed')
    pdf_path = os.path.join('data', 'raw', 'docu-1.pdf')

    try:
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
        logger.info(f"Processing document: {pdf_path}")
        result = doc_processor.process_document(pdf_path)

        # Print results
        if result['status'] == 'success':
            print("\n" + "="*50)
            print("PROCESSING SUMMARY")
            print("="*50)
            print(f"Document: {os.path.basename(pdf_path)}")
            print(f"Processing Time: {result['processing_time']:.2f} seconds")
            print(f"Fields Extracted: {result['statistics']['num_fields']}")
            print(f"Tables Detected: {result['statistics']['num_tables']}")
            print(f"Tokens Generated: {result['statistics']['num_tokens']}")
            print(f"Words Processed: {result['statistics']['num_words']}")
            print(f"Output Directory: {output_dir}")
            print("="*50 + "\n")
            print("Processing completed successfully! 🎉")
        else:
            print(f"\nProcessing failed: {result.get('message', 'Unknown error')} ❌")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"\nTest failed: {str(e)} ❌")

if __name__ == "__main__":
    main() 