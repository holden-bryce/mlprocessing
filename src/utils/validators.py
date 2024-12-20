from typing import Dict, List, Optional, Union
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DocumentValidator:
    def __init__(self):
        self.po_pattern = re.compile(r'^[A-Z0-9-]{4,20}$')
        self.amount_pattern = re.compile(r'^\$?\d{1,3}(?:,\d{3})*\.?\d{0,2}$')
        
    def validate_po_number(self, po_number: str) -> bool:
        """Validate purchase order number format"""
        if not po_number:
            return False
        return bool(self.po_pattern.match(po_number))

    def validate_date(self, date_str: str) -> bool:
        """Validate date format and logical value"""
        try:
            # Support multiple date formats
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%m-%d-%Y']:
                try:
                    datetime.strptime(date_str, fmt)
                    return True
                except ValueError:
                    continue
            return False
        except Exception as e:
            logger.error(f"Date validation error: {e}")
            return False

    def validate_amount(self, amount: str) -> bool:
        """Validate currency amount format"""
        if not amount:
            return False
        # Remove currency symbols and spaces
        clean_amount = amount.replace('$', '').replace(' ', '')
        return bool(self.amount_pattern.match(clean_amount))

    def validate_table_structure(self, table: Dict) -> bool:
        """Validate table structure and content"""
        if not table.get('cells') or not table.get('bbox'):
            return False

        # Check if table has consistent number of columns
        rows = self._group_cells_into_rows(table['cells'])
        if not rows:
            return False

        col_count = len(rows[0])
        return all(len(row) == col_count for row in rows)

    def _group_cells_into_rows(self, cells: List[Dict]) -> List[List[Dict]]:
        """Group table cells into rows based on vertical position"""
        if not cells:
            return []

        # Sort cells by y-coordinate
        sorted_cells = sorted(cells, key=lambda x: (x['bbox'][1], x['bbox'][0]))
        
        rows = []
        current_row = []
        current_y = sorted_cells[0]['bbox'][1]
        
        for cell in sorted_cells:
            if abs(cell['bbox'][1] - current_y) > 10:  # Threshold for new row
                if current_row:
                    rows.append(sorted(current_row, key=lambda x: x['bbox'][0]))
                current_row = [cell]
                current_y = cell['bbox'][1]
            else:
                current_row.append(cell)
                
        if current_row:
            rows.append(sorted(current_row, key=lambda x: x['bbox'][0]))
            
        return rows 