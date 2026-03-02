import os
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_name='yolo11s.pt'):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', model_name)
        self.model = YOLO(model_path)

        self.interesting_classes = [
            'person', 'car', 'door', 'tree', 'chair', 'table',
            'tv', 'laptop', 'cell phone', 'book',
            'stop sign', 'traffic sign','board'
        ]

        self.text_objects = [
            'book', 'tv', 'laptop', 'stop sign', 'traffic sign'
        ]

    def analyze_frame(self, frame):
        results = self.model(frame, conf=0.3, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                label = r.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                detections.append({
                    "label": label,
                    "box": [x1, y1, x2, y2]
                })

        return detections  

