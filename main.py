from core.ocr_engine import TextDetector
from hardware.camera_stream import CameraStream
from core.detector import ObjectDetector
from core.processor import HazardProcessor
from hardware.audio_output import AudioOutput

import cv2
import time

def main():
    # Initialize components
    audio = AudioOutput()
    # Change source to "0" for live camera or use the path to your image
    cap = CameraStream(source="data/street.png") 
    detector = ObjectDetector(model_name='models/yolo11s.pt')
    processor = HazardProcessor(cam_height=1.5, focal_len=2000)
    ocr = TextDetector(languages=['en'])

    print("Système d'assistance démarré... Appuyez sur 'q' pour quitter.")

    last_ocr_time = 0
    OCR_COOLDOWN = 5  # seconds between full scans

    while True:
        frame = cap.get_frame()
        if frame is None:
            break

        #Object detection
        detections = detector.analyze_frame(frame)
        
        # OCR detection 
        text_results = []
        if (time.time() - last_ocr_time) > OCR_COOLDOWN:
            text_results = ocr.detect_text(frame)
            last_ocr_time = time.time()

        scene_narration = ""

        if detections:
            counts = {}
            positions = {} 
            distances = {}
            object_texts = {} # Links text to specific objects 

            for d in detections:
                label = d['label']
                x1, y1, x2, y2 = map(int, d["box"])
                
                # Calculate distance and position
                dist = processor.get_distance(y2)
                pos = processor.get_position(x1, x2)
                
                # Logic to link OCR text to this specific object's bounding box
                associated_text = []
                for t in text_results:
                    tx1, ty1, tx2, ty2 = map(int, t["box"])
                    # Check if the text box is inside the object box
                    if x1 <= tx1 <= x2 and y1 <= ty1 <= y2:
                        associated_text.append(t["text"])
                
                if associated_text:
                    # Clean up common OCR artifacts found in your street image
                    clean_txt = " ".join(associated_text).replace("SC", "School Zone")
                    object_texts[label] = clean_txt

                # Group objects for narration
                counts[label] = counts.get(label, 0) + 1
                positions[label] = pos
                if label not in distances or dist < distances[label]:
                    distances[label] = dist

            # Build the narration phrases
            object_phrases = []
            for label, count in counts.items():
                dist_val = distances[label]
                pos_val = positions[label]
                
                # Special case: Narrate the text found on signs
                if label in object_texts:
                    phrase = f"a {label} that says {object_texts[label]} at {dist_val} meters {pos_val}"
                elif count > 1:
                    phrase = f"{count} {label}s starting at {dist_val} meters {pos_val}"
                else:
                    phrase = f"a {label} at {dist_val} meters {pos_val}"
                
                object_phrases.append(phrase)

            scene_narration = "In this scene, I see " + ", ".join(object_phrases) + "."
            
            # Add leftover text (like road markings not attached to an object)
            leftover_phrases = []
            for t in text_results:
                # If this text wasn't inside any object box, add it here
                tx1, ty1, tx2, ty2 = map(int, t["box"])
                is_leftover = True
                for d in detections:
                    dx1, dy1, dx2, dy2 = map(int, d["box"])
                    if dx1 <= tx1 <= dx2 and dy1 <= ty1 <= dy2:
                        is_leftover = False
                        break
                if is_leftover:
                    leftover_phrases.append(t['text'])

            if leftover_phrases:
                clean_leftover = " ".join(leftover_phrases).replace("SC", "School Zone")
                scene_narration += f" I also read text that says: {clean_leftover}."

        # Speak and Print the full Narrative
        if scene_narration:
            print("[NARRATOR]:", scene_narration)
            audio.speak(scene_narration)

        # Visualization (drawing on the frame)
        for d in detections:
            x1, y1, x2, y2 = map(int, d["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, d['label'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame
        resized = cv2.resize(frame, (800, int(frame.shape[0] * 800 / frame.shape[1])))
        cv2.imshow("Assistive View", resized)

        # Handle ending based on input type (image vs camera)
        if cap.is_image:
            print("Analysis complete. Press any key to close after audio finishes.")
            cv2.waitKey(0) # Wait for user to see the image
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()