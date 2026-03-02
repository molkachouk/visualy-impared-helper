from core.detector import ObjectDetector
from core.processor import HazardProcessor
from hardware.camera_stream import CameraStream
import cv2

def test_image():
    # 1. Load a static image
    source_path = "data/simple-living-room-with-wardrobe-desk.jpg"
    stream = CameraStream(source=source_path)
    
    # Ensure this matches your filename in /models
    detector = ObjectDetector(model_name='yolo11s.pt') 
    processor = HazardProcessor(cam_height=1.5) 

    frame = stream.get_frame()
    if frame is None:
        print(f"Error: Could not load image at {source_path}")
        return

    # Crucial: Sync the processor with the actual image dimensions
    processor.H_px, processor.W = frame.shape[:2]

    # 2. Run AI - Use the method that returns detailed analysis
    detections = detector.analyze_frame(frame)
    # The processor converts those detections into human-readable strings
    alerts = processor.process(detections)

    # 3. Draw and Show
    # 'alerts' is now a list of dictionaries from the updated Processor
    for i, alert_data in enumerate(alerts):
        # We get coordinates from the detections list
        x1, y1, x2, y2 = map(int, detections[i]['box'])
        
        # Draw Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Extract the text message from the alert dictionary
        msg = alert_data['text'] 
        
        cv2.putText(frame, msg, (x1, int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f"Analysis: {msg}")

        scale_percent = 800 / frame.shape[1]
        dim = (800, int(frame.shape[0] * scale_percent))
        resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)  
        cv2.imshow("Jetson View", resized_frame)
    print("Click on the image window and press any key to exit.")
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_image()