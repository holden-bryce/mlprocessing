import logging
from typing import List, Dict
import json
import os
from datetime import datetime
import re

class TableDetector:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.table_log_dir = os.path.join(output_dir, 'table_detection_logs')
        os.makedirs(self.table_log_dir, exist_ok=True)

    def detect_tables(self, textract_data: Dict) -> Dict:
        """
        Process tables from AWS Textract output
        """
        try:
            blocks = textract_data.get('Blocks', [])
            tables = []
            line_items = []

            # Find all table blocks
            table_blocks = [block for block in blocks if block['BlockType'] == 'TABLE']
            
            for table_block in table_blocks:
                table = self._process_table_block(table_block, blocks)
                if table:
                    tables.append(table)
                    # Check if this is a line item table
                    if self._is_line_item_table(table['headers']):
                        items = self._process_line_items(table)
                        line_items.extend(items)

            results = {
                'tables': tables,
                'line_items': line_items
            }
            
            self._log_table_results(results)
            return results

        except Exception as e:
            self.logger.error(f"Error detecting tables: {str(e)}")
            raise

    def _process_table_block(self, table_block: Dict, blocks: List[Dict]) -> Dict:
        """
        Process a single table block from Textract
        """
        try:
            # Get table cells using relationships
            cells = []
            rows = []
            current_row = []
            current_row_num = 0

            # Get all cells in the table
            for relationship in table_block.get('Relationships', []):
                if relationship['Type'] == 'CHILD':
                    for cell_id in relationship['Ids']:
                        cell_block = next((b for b in blocks if b['Id'] == cell_id), None)
                        if cell_block:
                            row_index = cell_block['RowIndex']
                            col_index = cell_block['ColumnIndex']
                            text = self._get_cell_text(cell_block, blocks)
                            cells.append({
                                'text': text,
                                'row': row_index,
                                'col': col_index
                            })

            # Sort cells by row and column
            cells.sort(key=lambda x: (x['row'], x['col']))

            # Group cells into rows
            current_row = []
            current_row_num = 1
            for cell in cells:
                if cell['row'] > current_row_num:
                    if current_row:
                        rows.append([c['text'] for c in current_row])
                    current_row = []
                    current_row_num = cell['row']
                current_row.append(cell)
            
            if current_row:
                rows.append([c['text'] for c in current_row])

            # First row is headers
            headers = rows[0] if rows else []
            data_rows = rows[1:] if len(rows) > 1 else []

            return {
                'headers': headers,
                'rows': data_rows,
                'num_columns': len(headers),
                'num_rows': len(data_rows),
                'confidence': table_block.get('Confidence', 0)
            }

        except Exception as e:
            self.logger.error(f"Error processing table block: {str(e)}")
            return None

    def _get_cell_text(self, cell_block: Dict, blocks: List[Dict]) -> str:
        """
        Get text content from a cell block
        """
        text = cell_block.get('Text', '')
        if not text and 'Relationships' in cell_block:
            for relationship in cell_block['Relationships']:
                if relationship['Type'] == 'CHILD':
                    words = [
                        block['Text']
                        for block in blocks
                        if block['Id'] in relationship['Ids']
                        and block['BlockType'] == 'WORD'
                    ]
                    text = ' '.join(words)
        return text.strip()

    def _is_line_item_table(self, headers: List[str]) -> bool:
        """
        Determine if table contains line items
        """
        headers_str = ' '.join(str(h).lower() for h in headers)
        return any([
            'description' in headers_str,
            'material' in headers_str,
            'product' in headers_str,
            'item' in headers_str,
            'part' in headers_str,
            'service' in headers_str
        ])

    def _process_line_items(self, table: Dict) -> List[Dict]:
        """
        Process line items from a table
        """
        try:
            headers = table['headers']
            rows = table['rows']
            header_map = self._create_header_map(headers)
            line_items = []

            for row in rows:
                if not self._is_summary_row(row):
                    item = self._extract_line_item(row, header_map)
                    if item and item['description']:
                        line_items.append(item)

            return line_items
        except Exception as e:
            self.logger.error(f"Error processing line items: {str(e)}")
            return []

    def _create_header_map(self, headers: List[str]) -> Dict:
        """
        Create mapping of column types based on header names
        """
        header_map = {}
        headers_str = [str(h).lower() for h in headers]
        
        for i, header in enumerate(headers_str):
            # Description/Product/Material
            if any(term in header for term in ['desc', 'product', 'material', 'item', 'part', 'service']):
                header_map['description'] = i
            
            # Quantity
            elif any(term in header for term in ['qty', 'quant', 'number', 'units']):
                header_map['quantity'] = i
            
            # Price/Rate/Cost
            elif any(term in header for term in ['price', 'rate', 'amount', 'cost']):
                if 'unit' in header:
                    header_map['unit_price'] = i
                elif 'total' in header:
                    header_map['total_price'] = i
                else:
                    header_map['price'] = i
            
            # UoM
            elif any(term in header for term in ['uom', 'unit', 'measure']):
                header_map['uom'] = i
            
            # Date
            elif any(term in header for term in ['date', 'delivery', 'schedule']):
                header_map['date'] = i

        return header_map

    def _is_summary_row(self, row: List[str]) -> bool:
        """
        Check if row is a summary row
        """
        row_str = ' '.join(str(cell).lower() for cell in row)
        return any(term in row_str for term in [
            'total', 'subtotal', '**', 'sum', 'amount due', 
            'balance', 'tax', 'shipping'
        ])

    def _extract_line_item(self, row: List[str], header_map: Dict) -> Dict:
        """
        Extract line item from a row
        """
        try:
            item = {
                'description': '',
                'quantity': 0,
                'unit_price': 0,
                'total_price': 0,
                'uom': '',
                'date': ''
            }

            for field, index in header_map.items():
                if index < len(row):
                    value = row[index]
                    if field in ['quantity', 'unit_price', 'total_price', 'price']:
                        item[field] = self._extract_number(value)
                    else:
                        item[field] = value.strip()

            # Calculate total price if missing
            if not item['total_price'] and item['quantity'] and item['unit_price']:
                item['total_price'] = item['quantity'] * item['unit_price']

            return item
        except Exception as e:
            self.logger.error(f"Error extracting line item: {str(e)}")
            return None

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

    def _log_table_results(self, results: Dict):
        """
        Log table detection results
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'table_detection_log_{timestamp}.json'
            log_path = os.path.join(self.table_log_dir, log_filename)
            
            log_data = {
                'timestamp': timestamp,
                'num_tables': len(results['tables']),
                'num_line_items': len(results['line_items']),
                'results': results
            }
            
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2)
            
            self.logger.info(f"Table detection results logged to {log_path}")
            
        except Exception as e:
            self.logger.error(f"Error logging table results: {str(e)}") 