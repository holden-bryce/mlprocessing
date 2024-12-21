import boto3
from typing import Dict, List
import logging
import os
import json
from datetime import datetime
from botocore.exceptions import ClientError

class OCRProcessor:
    def __init__(self, output_dir: str):
        """
        Initialize OCR Processor with AWS Textract
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Initialize AWS clients
        self.textract = boto3.client('textract')
        self.s3 = boto3.client('s3')
        
        # Ensure OCR log directory exists
        self.ocr_log_dir = os.path.join(output_dir, 'ocr_logs')
        os.makedirs(self.ocr_log_dir, exist_ok=True)

    def extract_text(self, pdf_path: str) -> Dict:
        """
        Extract text and form data using AWS Textract
        """
        try:
            # Read PDF file
            with open(pdf_path, 'rb') as document:
                file_bytes = document.read()

            # Call Textract with all analysis features
            response = self.textract.analyze_document(
                Document={'Bytes': file_bytes},
                FeatureTypes=['TABLES', 'FORMS', 'SIGNATURES']
            )

            # Process Textract response
            blocks = response['Blocks']
            
            # Extract all components
            processed_data = {
                'raw_text': self._extract_raw_text(blocks),
                'lines': self._extract_lines(blocks),
                'words': self._extract_words(blocks),
                'forms': self._extract_forms(blocks),
                'tables': self._extract_tables(blocks),
                'signatures': self._extract_signatures(blocks),
                'Blocks': blocks  # Include original blocks for downstream processing
            }

            # Log results
            self._log_ocr_results(processed_data, pdf_path)
            
            return processed_data

        except ClientError as e:
            error_msg = f"AWS Textract error: {str(e)}"
            self.logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Error in text extraction: {str(e)}"
            self.logger.error(error_msg)
            raise

    def _extract_raw_text(self, blocks: List[Dict]) -> str:
        """
        Extract raw text from Textract blocks
        """
        text_blocks = [
            block['Text'] 
            for block in blocks 
            if block['BlockType'] == 'LINE'
        ]
        return ' '.join(text_blocks)

    def _extract_lines(self, blocks: List[Dict]) -> List[str]:
        """
        Extract lines of text from Textract blocks
        """
        return [
            block['Text']
            for block in blocks
            if block['BlockType'] == 'LINE'
        ]

    def _extract_words(self, blocks: List[Dict]) -> List[str]:
        """
        Extract individual words from Textract blocks
        """
        return [
            block['Text']
            for block in blocks
            if block['BlockType'] == 'WORD'
        ]

    def _extract_forms(self, blocks: List[Dict]) -> List[Dict]:
        """
        Extract form fields (key-value pairs) from Textract blocks
        """
        forms = []
        for block in blocks:
            if block['BlockType'] == 'KEY_VALUE_SET':
                if 'KEY' in block.get('EntityTypes', []):
                    key = self._get_text_from_relationships(block, blocks)
                    value_block = self._find_value_block(block, blocks)
                    if value_block:
                        value = self._get_text_from_relationships(value_block, blocks)
                        if key and value:
                            forms.append({
                                'key': key.strip(),
                                'value': value.strip(),
                                'confidence': block.get('Confidence', 0)
                            })
        return forms

    def _extract_tables(self, blocks: List[Dict]) -> List[Dict]:
        """
        Extract table structures from Textract blocks
        """
        tables = []
        for block in blocks:
            if block['BlockType'] == 'TABLE':
                table_data = {
                    'rows': block.get('RowCount', 0),
                    'columns': block.get('ColumnCount', 0),
                    'confidence': block.get('Confidence', 0),
                    'cells': self._get_table_cells(block, blocks)
                }
                tables.append(table_data)
        return tables

    def _extract_signatures(self, blocks: List[Dict]) -> List[Dict]:
        """
        Extract signature information from Textract blocks
        """
        signatures = []
        for block in blocks:
            if block['BlockType'] == 'SIGNATURE':
                signature_data = {
                    'confidence': block.get('Confidence', 0),
                    'geometry': block.get('Geometry', {}),
                    'id': block.get('Id', '')
                }
                signatures.append(signature_data)
        return signatures

    def _get_text_from_relationships(self, block: Dict, blocks: List[Dict]) -> str:
        """
        Get text from block relationships
        """
        text = []
        if 'Relationships' in block:
            for relationship in block['Relationships']:
                if relationship['Type'] == 'CHILD':
                    for child_id in relationship['Ids']:
                        child_block = next(
                            (b for b in blocks if b['Id'] == child_id),
                            None
                        )
                        if child_block and child_block['BlockType'] == 'WORD':
                            text.append(child_block['Text'])
        return ' '.join(text)

    def _find_value_block(self, key_block: Dict, blocks: List[Dict]) -> Dict:
        """
        Find the value block associated with a key block
        """
        for relationship in key_block.get('Relationships', []):
            if relationship['Type'] == 'VALUE':
                value_id = relationship['Ids'][0]
                return next(
                    (b for b in blocks if b['Id'] == value_id),
                    None
                )
        return None

    def _get_table_cells(self, table_block: Dict, blocks: List[Dict]) -> List[List[str]]:
        """
        Extract cells from a table block
        """
        cells = []
        if 'Relationships' in table_block:
            for relationship in table_block['Relationships']:
                if relationship['Type'] == 'CHILD':
                    row = []
                    for cell_id in relationship['Ids']:
                        cell_block = next(
                            (b for b in blocks if b['Id'] == cell_id),
                            None
                        )
                        if cell_block:
                            cell_text = self._get_text_from_relationships(cell_block, blocks)
                            row.append(cell_text)
                    if row:
                        cells.append(row)
        return cells

    def _log_ocr_results(self, processed_data: Dict, source_file: str):
        """
        Log OCR results
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.basename(source_file)
            log_filename = f'ocr_log_{filename}_{timestamp}.json'
            log_path = os.path.join(self.ocr_log_dir, log_filename)
            
            # Create log data with metadata
            log_data = {
                'timestamp': timestamp,
                'source_file': filename,
                'statistics': {
                    'num_words': len(processed_data['words']),
                    'num_lines': len(processed_data['lines']),
                    'num_forms': len(processed_data['forms']),
                    'num_tables': len(processed_data['tables']),
                    'num_signatures': len(processed_data['signatures'])
                },
                'results': {
                    'text': processed_data['raw_text'],
                    'forms': processed_data['forms'],
                    'tables': processed_data['tables'],
                    'signatures': processed_data['signatures']
                }
            }
            
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2)
            
            self.logger.info(f"OCR results logged to {log_path}")
            
        except Exception as e:
            self.logger.error(f"Error logging OCR results: {str(e)}")