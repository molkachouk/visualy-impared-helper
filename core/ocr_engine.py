import easyocr
import cv2
import numpy as np

class TextDetector:
    def __init__(self, languages=['en']):
        self.reader = easyocr.Reader(languages)

    def detect_text(self, frame, boxes=None):
        texts = []

        if boxes and len(boxes) > 0:
            # OCR inside detected objects
            for x1, y1, x2, y2 in boxes:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                texts += self._ocr_crop(crop, x1, y1)
        else:
            # Fallback: detect text regions automatically
            texts += self._fallback_text_regions(frame)

        # Debug: print all detected texts
        if texts:
            print("OCR DETECTED TEXTS:")
            for t in texts:
                print("  ", t["text"], "->", t["box"])
        else:
            print("OCR detected no text in this frame.")

        return texts

    def _ocr_crop(self, crop, offset_x, offset_y):
        results = []
        ocr = self.reader.readtext(crop)
        for bbox, text, conf in ocr:
            if conf > 0.5:
                # bbox coordinates relative to crop
                (tl, tr, br, bl) = bbox
                x1 = int(tl[0] + offset_x)
                y1 = int(tl[1] + offset_y)
                x2 = int(br[0] + offset_x)
                y2 = int(br[1] + offset_y)
                results.append({
                    "text": text[:40],
                    "box": [x1, y1, x2, y2]
                })
        return results

    def _fallback_text_regions(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use Canny edge detection instead of adaptiveThreshold
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges to connect text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        texts = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < 30 or h < 15:  # skip too small boxes
                continue

            crop = frame[y:y+h, x:x+w]
            results = self._ocr_crop(crop, x, y)

            # Debug: print detected text in each region
            if results:
                print(f"OCR found {len(results)} texts in region x:{x}, y:{y}, w:{w}, h:{h}")

            texts += results

        if not texts:
            print("Fallback OCR found nothing!")

        return texts

