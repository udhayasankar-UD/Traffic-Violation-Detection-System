import cv2
import math
import os
import csv
import re
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import easyocr
import numpy as np
import argparse


MODEL_PATH = 'runs/detect/yolov8n_helmet_custom2/weights/best.pt'
CONFIDENCE_THRESH = 0.4
DISPLAY_WIDTH = 780
MAX_AGE = 50
ASSOCIATION_THRESHOLD = 500 

EVIDENCE_DIR = 'violations'
LOG_FILE = os.path.join(EVIDENCE_DIR, 'log.csv')

os.makedirs(EVIDENCE_DIR, exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Plate Number', 'Image Filename'])

print("Initializing EasyOCR reader...")
reader = easyocr.Reader(['en'], gpu=True)
print("EasyOCR reader initialized.")

COLORS = {
    'Helmet': (0, 255, 0), 'Non-helmet': (0, 0, 255), 'Rider': (255, 255, 0),
    'licence_plate': (0, 255, 255), 'Non_rider': (128, 0, 128),
}

def clean_plate_text(text):
    return re.sub(r'[^A-Z0-9]', '', text).upper()

def get_center(box):
    x1, y1, x2, y2 = map(int, box)
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def ocr_on_plate(plate_crop, reader):
    h, w, _ = plate_crop.shape
    large_plate = cv2.resize(plate_crop, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(large_plate, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    strategy1_img = clahe.apply(gray)
    _, strategy2_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    images_to_try = [strategy1_img, strategy2_img, gray]
    best_text = None
    highest_confidence = 0.0

    for i, img in enumerate(images_to_try):
        ocr_result = reader.readtext(img, detail=1, paragraph=False, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        if ocr_result:
            for (bbox, text, conf) in ocr_result:
                if conf > highest_confidence:
                    cleaned = clean_plate_text(text)
                    if cleaned and len(cleaned) > 6:
                        highest_confidence = conf
                        best_text = cleaned
                        print(f"  OCR Strategy {i+1} found new best: '{best_text}' with confidence {conf:.2f}")

    return best_text


def process_single_image(image_path, model, reader):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image file {image_path}"); return

    results = model(frame, verbose=False)[0]
    
    violators, plates, all_detections = [], [], []

    for box in results.boxes:
        if box.conf[0] > CONFIDENCE_THRESH:
            detection_data = {'box': box.xyxy[0].tolist(), 'class': model.names[int(box.cls[0])]}
            all_detections.append(detection_data)
            if detection_data['class'] == 'Non-helmet': violators.append(detection_data)
            elif detection_data['class'] == 'licence_plate': plates.append(detection_data)
    
    for plate in plates:
        x1, y1, x2, y2 = map(int, plate['box'])
        if x1 < x2 and y1 < y2:
            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size > 0:
                plate_text = ocr_on_plate(plate_crop, reader)
                if plate_text: plate['text'] = plate_text
    
    saved_this_image = False
    for violator in violators:
        if saved_this_image: break
        violator_center = get_center(violator['box'])
        
        closest_plate = None
        min_dist = float('inf')

        for plate in plates:
            if 'text' in plate:
                dist = math.dist(violator_center, get_center(plate['box']))
                print(f"DIAGNOSTIC: Distance from violator to plate '{plate['text']}' is {dist:.2f} pixels.")
                if dist < min_dist: min_dist, closest_plate = dist, plate
        
        if closest_plate and min_dist < ASSOCIATION_THRESHOLD:
            plate_text = closest_plate['text']
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{plate_text}_{timestamp}.jpg"
            filepath = os.path.join(EVIDENCE_DIR, filename)

            print(f"\n--- VIOLATION DETECTED! ---\nPlate: {plate_text}\nSaving to: {filepath}\n----------------------")
            with open(LOG_FILE, 'a', newline='') as f:
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), plate_text, filename])
            
            drawn_frame = frame.copy()
            for det in all_detections:
                b, cls = det['box'], det['class']
                x1, y1, x2, y2 = map(int, b)
                color = (0,0,255) if cls == 'Non-helmet' else COLORS.get(cls, (128,128,128))
                label = cls
                if cls == 'Non-helmet': label += " [VIOLATION]"
                for p in plates:
                    if p['box'] == b and 'text' in p: label += f" [{p['text']}]"
                cv2.rectangle(drawn_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(drawn_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imwrite(filepath, drawn_frame)
            print(f"--- EVIDENCE IMAGE SAVED: {filepath} ---")
            saved_this_image = True
    
    display_frame = frame.copy()
    for det in all_detections:
        b, cls = det['box'], det['class']
        x1, y1, x2, y2 = map(int, b)
        color = (0,0,255) if cls == 'Non-helmet' else COLORS.get(cls, (128,128,128))
        label = cls
        if cls == 'Non-helmet': label += " [VIOLATION]"
        for p in plates:
            if p['box'] == b and 'text' in p: label += f" [{p['text']}]"
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    display_frame_resized = cv2.resize(display_frame, (DISPLAY_WIDTH, int(display_frame.shape[0] * DISPLAY_WIDTH / display_frame.shape[1])))
    cv2.imshow("Violation Detection", display_frame_resized)
    print("Press any key to exit.")
    cv2.waitKey(0)

def main():
    parser = argparse.ArgumentParser(description="Traffic Violation Detection")
    parser.add_argument("--video", type=str, help="Path to the video file.")
    parser.add_argument("--image", type=str, help="Path to the image file.")
    args = parser.parse_args()

    model = YOLO(MODEL_PATH)

    if args.video:
        print("Video processing is not implemented in this version. Please use --image.")
    elif args.image:
        process_single_image(args.image, model, reader)
    else:
        print("Please provide a source. Use --video <path> or --image <path>.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()