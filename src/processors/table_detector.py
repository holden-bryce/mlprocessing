import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
import logging
from dataclasses import dataclass

@dataclass
class TableCell:
    text: str
    bbox: List[int]
    confidence: float
    row_idx: int
    col_idx: int

@dataclass
class Table:
    cells: List[TableCell]
    bbox: List[int]
    num_rows: int
    num_cols: int
    headers: List[str]

class TableDetector:
    def __init__(self, 
                 min_confidence: float = 30.0,
                 line_min_length: int = 50,
                 line_threshold: int = 30):
        self.min_confidence = min_confidence
        self.line_min_length = line_min_length
        self.line_threshold = line_threshold
        self.logger = logging.getLogger(__name__)

    def detect_tables(self, 
                     image: np.ndarray, 
                     ocr_data: Dict) -> List[Table]:
        """Detect tables in document image"""
        try:
            # Detect table regions
            table_regions = self._detect_table_regions(image)
            
            # Extract cells from each table region
            tables = []
            for region in table_regions:
                table = self._extract_table_structure(
                    region, 
                    ocr_data
                )
                if table:
                    tables.append(table)
            
            return tables

        except Exception as e:
            self.logger.error(f"Error detecting tables: {str(e)}")
            return []

    def _detect_table_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect table regions using line detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(
            gray, 
            0, 
            255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Detect lines
        horizontal = self._detect_lines(binary, 'horizontal')
        vertical = self._detect_lines(binary, 'vertical')

        # Combine lines
        table_mask = cv2.bitwise_or(horizontal, vertical)
        
        # Find table regions
        contours, _ = cv2.findContours(
            table_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Extract regions
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > self.line_min_length and h > self.line_min_length:
                regions.append([x, y, x + w, y + h])

        return regions

    def _detect_lines(self, 
                     binary: np.ndarray, 
                     direction: str) -> np.ndarray:
        """Detect lines in specific direction"""
        if direction == 'horizontal':
            kernel_length = binary.shape[1] // 40
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        else:
            kernel_length = binary.shape[0] // 40
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

        # Apply morphology
        detected = cv2.erode(binary, kernel, iterations=1)
        detected = cv2.dilate(detected, kernel, iterations=1)

        return detected

    def _extract_table_structure(self, 
                               region: List[int], 
                               ocr_data: Dict) -> Optional[Table]:
        """Extract table structure from region"""
        # Filter OCR results within region
        cells = self._get_cells_in_region(region, ocr_data)
        if not cells:
            return None

        # Group cells into rows and columns
        rows = self._group_cells_into_rows(cells)
        if not rows:
            return None

        # Identify headers
        headers = self._identify_headers(rows[0])

        return Table(
            cells=cells,
            bbox=region,
            num_rows=len(rows),
            num_cols=len(rows[0]),
            headers=headers
        )

    def _get_cells_in_region(self, 
                            region: List[int], 
                            ocr_data: Dict) -> List[TableCell]:
        """Get cells within table region"""
        cells = []
        for i in range(len(ocr_data["text"])):
            bbox = ocr_data["boxes"][i]
            if self._is_within_region(bbox, region):
                cells.append(
                    TableCell(
                        text=ocr_data["text"][i],
                        bbox=bbox,
                        confidence=ocr_data["confidence"][i],
                        row_idx=-1,  # Will be set later
                        col_idx=-1   # Will be set later
                    )
                )
        return cells

    def _is_within_region(self, 
                         bbox: List[int], 
                         region: List[int]) -> bool:
        """Check if bbox is within region"""
        return (bbox[0] >= region[0] and bbox[2] <= region[2] and
                bbox[1] >= region[1] and bbox[3] <= region[3])

    def _group_cells_into_rows(self, 
                              cells: List[TableCell]) -> List[List[TableCell]]:
        """Group cells into rows based on vertical position"""
        # Sort cells by y-coordinate
        sorted_cells = sorted(cells, key=lambda x: (x.bbox[1], x.bbox[0]))
        
        rows = []
        current_row = []
        current_y = sorted_cells[0].bbox[1]
        
        for cell in sorted_cells:
            if abs(cell.bbox[1] - current_y) > 10:  # Threshold for new row
                if current_row:
                    # Sort cells in row by x-coordinate
                    current_row.sort(key=lambda x: x.bbox[0])
                    rows.append(current_row)
                current_row = [cell]
                current_y = cell.bbox[1]
            else:
                current_row.append(cell)
        
        if current_row:
            current_row.sort(key=lambda x: x.bbox[0])
            rows.append(current_row)
        
        # Set row and column indices
        for row_idx, row in enumerate(rows):
            for col_idx, cell in enumerate(row):
                cell.row_idx = row_idx
                cell.col_idx = col_idx
        
        return rows

    def _identify_headers(self, row: List[TableCell]) -> List[str]:
        """Identify table headers from first row"""
        return [cell.text for cell in row] 