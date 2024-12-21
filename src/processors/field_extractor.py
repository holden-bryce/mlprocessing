import logging
from typing import Dict, List
import json
import os
from datetime import datetime
import re

class FieldExtractor:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.field_log_dir = os.path.join(output_dir, 'field_extraction_logs')
        os.makedirs(self.field_log_dir, exist_ok=True)

    def extract_fields_and_labels(self, textract_data: Dict) -> Dict:
        """
        Extract fields from AWS Textract output
        """
        try:
            blocks = textract_data.get('Blocks', [])
            
            # Extract key-value pairs from Textract blocks
            key_value_pairs = self._extract_key_value_pairs(blocks)
            
            # Extract structured information
            structured_data = self._extract_document_info(key_value_pairs)
            
            self._log_extraction_results(structured_data)
            return structured_data
            
        except Exception as e:
            self.logger.error(f"Error extracting fields: {str(e)}")
            raise

    def _extract_key_value_pairs(self, blocks: List[Dict]) -> Dict:
        """
        Extract key-value pairs from Textract blocks
        """
        key_value_pairs = {}
        
        # Find all key-value set blocks
        for block in blocks:
            if block['BlockType'] == 'KEY_VALUE_SET':
                if 'KEY' in block['EntityTypes']:
                    key = self._get_text_from_relationships(block, blocks)
                    value_block = self._find_value_block(block, blocks)
                    if value_block:
                        value = self._get_text_from_relationships(value_block, blocks)
                        if key and value:
                            key_value_pairs[key.strip()] = value.strip()

        return key_value_pairs

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

    def _extract_document_info(self, key_value_pairs: Dict) -> Dict:
        """
        Extract structured document information from key-value pairs
        """
        return {
            "document_info": {
                "po_number": self._extract_field(key_value_pairs, [
                    'PO Number', 'P.O.Number', 'Order Number', 'Purchase Order',
                    'PO No', 'PO NO.', 'PO#'
                ]),
                "order_date": self._extract_field(key_value_pairs, [
                    'Order Date', 'Invoice Date', 'Ordered', 'Date',
                    'PO Date', 'Purchase Order Date'
                ]),
                "document_type": self._determine_document_type(key_value_pairs),
                "currency": self._extract_currency(key_value_pairs)
            },
            "vendor_info": {
                "name": self._extract_field(key_value_pairs, [
                    'Vendor Name', 'Ordered From', 'Supplier', 'Seller',
                    'Company Name', 'From'
                ]),
                "number": self._extract_field(key_value_pairs, [
                    'Vendor #', 'Vendor Number', 'Supplier ID', 'Vendor ID',
                    'Supplier Number', 'Vendor Code'
                ]),
                "address": self._extract_field(key_value_pairs, [
                    'Billing Address', 'Vendor Address', 'From Address',
                    'Supplier Address', 'Bill From'
                ]),
                "tax_id": self._extract_field(key_value_pairs, [
                    'Tax ID', 'VAT Number', 'Tax Number', 'EIN',
                    'Federal ID Number', 'VAT Registration No'
                ]),
                "contact": self._extract_field(key_value_pairs, [
                    'Contact', 'Contact Person', 'Vendor Contact',
                    'Supplier Contact'
                ])
            },
            "shipping_info": {
                "address": self._extract_field(key_value_pairs, [
                    'Ship To', 'Shipping Address', 'Deliver To',
                    'Delivery Address', 'Ship To Address'
                ]),
                "method": self._extract_field(key_value_pairs, [
                    'Ship Via', 'Shipping Method', 'Delivery Method',
                    'Ship By', 'Incoterms'
                ]),
                "terms": self._extract_field(key_value_pairs, [
                    'Shipping Terms', 'Delivery Terms', 'Ship Terms'
                ])
            },
            "payment_info": {
                "terms": self._extract_field(key_value_pairs, [
                    'Payment Terms', 'Terms', 'Payment Method',
                    'Terms and Conditions'
                ]),
                "subtotal": self._extract_number(self._extract_field(key_value_pairs, [
                    'Subtotal', 'Net Amount', 'Amount Before Tax'
                ])),
                "tax_amount": self._extract_number(self._extract_field(key_value_pairs, [
                    'Tax', 'VAT', 'Sales Tax', 'GST'
                ])),
                "total_amount": self._extract_number(self._extract_field(key_value_pairs, [
                    'Total', 'Invoice Total', 'Total Amount', 'Grand Total',
                    'Amount Due', 'Total Order Value'
                ])),
                "currency": self._extract_currency(key_value_pairs)
            }
        }

    def _determine_document_type(self, key_value_pairs: Dict) -> str:
        """
        Determine the type of document based on key-value pairs
        """
        keywords = {
            'invoice': ['invoice', 'bill', 'tax invoice'],
            'purchase_order': ['purchase order', 'po', 'order'],
            'packing_slip': ['packing slip', 'delivery note', 'shipping list'],
            'quote': ['quote', 'quotation', 'estimate']
        }
        
        text = ' '.join(key_value_pairs.keys()).lower()
        for doc_type, terms in keywords.items():
            if any(term in text for term in terms):
                return doc_type
        return 'unknown'

    def _extract_currency(self, key_value_pairs: Dict) -> str:
        """
        Extract currency from amounts in key-value pairs
        """
        currency_patterns = {
            'USD': r'\$|USD|US Dollar',
            'EUR': r'€|EUR|Euro',
            'GBP': r'£|GBP|British Pound',
            'JPY': r'¥|JPY|Japanese Yen',
            'AUD': r'AUD|Australian Dollar',
            'CAD': r'CAD|Canadian Dollar'
        }
        
        text = ' '.join(key_value_pairs.values()).upper()
        for currency, pattern in currency_patterns.items():
            if re.search(pattern, text):
                return currency
        return ''

    def _extract_field(self, data: Dict, possible_keys: List[str]) -> str:
        """
        Extract field from multiple possible keys
        """
        for key in possible_keys:
            # Try exact match first
            if key in data:
                return data[key]
            
            # Try case-insensitive match
            for k in data.keys():
                if k.lower() == key.lower():
                    return data[k]
                
            # Try partial match
            for k in data.keys():
                if key.lower() in k.lower() or k.lower() in key.lower():
                    return data[k]
        return ''

    def _extract_number(self, value: str) -> float:
        """
        Extract number from string
        """
        try:
            if not value:
                return 0
            # Remove currency symbols and other non-numeric characters
            clean_value = re.sub(r'[^\d.,\-]', '', str(value))
            # Handle negative numbers
            multiplier = -1 if '-' in clean_value else 1
            # Remove thousands separators and convert to float
            clean_value = clean_value.replace(',', '')
            return float(clean_value) * multiplier
        except:
            return 0

    def _log_extraction_results(self, structured_data: Dict):
        """
        Log extraction results
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'field_extraction_log_{timestamp}.json'
            log_path = os.path.join(self.field_log_dir, log_filename)
            
            log_data = {
                'timestamp': timestamp,
                'results': structured_data
            }
            
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2)
            
            self.logger.info(f"Field extraction results logged to {log_path}")
            
        except Exception as e:
            self.logger.error(f"Error logging field results: {str(e)}")