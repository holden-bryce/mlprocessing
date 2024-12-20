import pytesseract
from PIL import Image
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from ..utils.normalizers import BBoxNormalizer

class OCRProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.bbox_normalizer = BBoxNormalizer()
        # Update with your Tesseract path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def process_image(self,
                     image_path: Union[str, Path, np.ndarray],
                     normalize_boxes: bool = True) -> Dict:
        """Process image with OCR.

        Args:
            image_path: Path to image file
            normalize_boxes: Whether to normalize bounding boxes

        Returns:
            Dict containing OCR results
        """
        try:
            # Load image
            if isinstance(image_path, (str, Path)):
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = image_path

            # Get OCR results
            ocr_data = pytesseract.image_to_data(
                image, 
                output_type=pytesseract.Output.DICT
            )

            # Extract text and boxes
            text = []
            boxes = []
            confidences = []

            for i in range(len(ocr_data['text'])):
                if ocr_data['conf'][i] > 0:  # Filter out low confidence
                    text.append(ocr_data['text'][i])
                    boxes.append([
                        ocr_data['left'][i],
                        ocr_data['top'][i],
                        ocr_data['left'][i] + ocr_data['width'][i],
                        ocr_data['top'][i] + ocr_data['height'][i]
                    ])
                    confidences.append(ocr_data['conf'][i])

            # Normalize boxes if requested
            if normalize_boxes:
                boxes = self.bbox_normalizer.normalize(
                    boxes,
                    image.shape[1],  # width
                    image.shape[0]   # height
                )

            return {
                'text': text,
                'boxes': boxes,
                'confidences': confidences,
                'image_size': (image.shape[1], image.shape[0])
            }

        except Exception as e:
            self.logger.error(f"OCR processing failed: {str(e)}")
            raise 