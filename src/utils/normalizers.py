from typing import List, Tuple, Dict
import numpy as np

class BBoxNormalizer:
    def __init__(self):
        pass

    def normalize(self, 
                 boxes: List[List[int]], 
                 width: int, 
                 height: int) -> List[List[float]]:
        """Normalize bounding box coordinates to [0, 1] range.
        
        Args:
            boxes: List of bounding boxes in format [x1, y1, x2, y2]
            width: Image width
            height: Image height
            
        Returns:
            List of normalized bounding boxes
        """
        normalized_boxes = []
        for box in boxes:
            # Ensure box coordinates are within image bounds
            x1 = max(0, min(box[0], width))
            y1 = max(0, min(box[1], height))
            x2 = max(0, min(box[2], width))
            y2 = max(0, min(box[3], height))
            
            # Normalize coordinates
            normalized_box = [
                x1 / width,
                y1 / height,
                x2 / width,
                y2 / height
            ]
            normalized_boxes.append(normalized_box)
            
        return normalized_boxes

    def denormalize(self, 
                   boxes: List[List[float]], 
                   width: int, 
                   height: int) -> List[List[int]]:
        """Convert normalized coordinates back to pixel coordinates.
        
        Args:
            boxes: List of normalized bounding boxes
            width: Image width
            height: Image height
            
        Returns:
            List of denormalized bounding boxes
        """
        denormalized_boxes = []
        for box in boxes:
            denormalized_box = [
                int(box[0] * width),
                int(box[1] * height),
                int(box[2] * width),
                int(box[3] * height)
            ]
            denormalized_boxes.append(denormalized_box)
            
        return denormalized_boxes

class TextNormalizer:
    def __init__(self):
        self.amount_chars = set('$,.0123456789')
        
    def normalize_amount(self, text: str) -> str:
        """Normalize currency amount"""
        # Keep only relevant characters
        clean_text = ''.join(c for c in text if c in self.amount_chars)
        
        # Remove multiple decimal points
        parts = clean_text.split('.')
        if len(parts) > 2:
            clean_text = parts[0] + '.' + parts[1]
            
        return clean_text

    def normalize_date(self, text: str) -> str:
        """Normalize date format to YYYY-MM-DD"""
        # Implementation depends on your specific needs
        return text 