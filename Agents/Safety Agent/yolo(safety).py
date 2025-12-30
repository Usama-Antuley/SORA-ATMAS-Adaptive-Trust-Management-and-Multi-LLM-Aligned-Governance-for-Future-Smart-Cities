# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 12:53:10 2025

@author: FMT COMPUTERS
"""

import torch
import cv2
import numpy as np
import os
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from ultralytics import YOLO
import threading
import time
import json
import datetime

# Configuration
GMAIL_ADDRESS = "usamatestphd@gmail.com"
GMAIL_APP_PASSWORD = "APP PASSWORD"  # Replace with your Gmail App Passwords
RECIPIENT_EMAIL = "waqas.arif.v@nu.edu.pk "

# Initial latitude and longitude for Karachi
LATITUDE = 24.8607
LONGITUDE = 67.0011

# Model and video paths
model_path = r"C:\Users\FMT COMPUTERS\Downloads\Code Paper 3\sPYDER\Real-Time-Smoke-Fire-Detection-YOLO11\models\kaggle developed models\yolo11-d-fire-dataset.pt"
video_path = r"C:\Users\FMT COMPUTERS\Downloads\Code Paper 3\sPYDER\yolov5-fire-detection\buildingonfire.mp4"
output_dir = r"C:\Users\FMT COMPUTERS\Downloads\Code Paper 3\sPYDER\yolov5-fire-detection"
output_path = os.path.join(output_dir, "output_fire.mp4")
alerts_json_path = os.path.join(output_dir, "alerts.json")

# Validate paths
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit(1)
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    sys.exit(1)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize alerts JSON if not exists
if not os.path.exists(alerts_json_path):
    with open(alerts_json_path, 'w') as f:
        json.dump([], f)

# Load YOLOv11 model
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading YOLOv11 model: {e}")
    sys.exit(1)

# Initialize video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    sys.exit(1)

# Video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Input video properties: Width={frame_width}, Height={frame_height}, FPS={fps}")

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
if not out.isOpened():
    print(f"Error: Could not open video writer at {output_path}")
    cap.release()
    sys.exit(1)

# Function to log alert to JSON
def log_alert_to_json(alert_details):
    try:
        with open(alerts_json_path, 'r+') as f:
            alerts = json.load(f)
            alerts.append(alert_details)
            f.seek(0)
            json.dump(alerts, f, indent=4)
        print("Alert logged to JSON successfully")
    except Exception as e:
        print(f"Error logging alert to JSON: {str(e)}")

# Gmail notification function
def send_gmail_alert(image, message, lat, long):
    if GMAIL_APP_PASSWORD == "your_app_password":
        print("Error: Gmail App Password not configured. Skipping email alert.")
        return
    try:
        enhanced_message = f"{message}\nLocation: Latitude {lat}, Longitude {long}"
        temp_image_path = os.path.join(output_dir, "temp_alert_image.jpg")
        if not cv2.imwrite(temp_image_path, image):
            raise Exception("Failed to save temporary image")
        
        msg = MIMEMultipart()
        msg['From'] = GMAIL_ADDRESS
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = "Flare Guard: Fire/Smoke Detection Alert"
        msg.attach(MIMEText(enhanced_message, 'plain'))
        with open(temp_image_path, 'rb') as img_file:
            img = MIMEImage(img_file.read(), name="detected_image.jpg")
            msg.attach(img)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())
        print("Gmail alert sent successfully")
        os.remove(temp_image_path)

        # Log to JSON
        alert_details = {
            "timestamp": datetime.datetime.now().isoformat(),
            "message": enhanced_message,
            "latitude": lat,
            "longitude": long
        }
        threading.Thread(target=log_alert_to_json, args=(alert_details,)).start()
    except Exception as e:
        print(f"Error sending Gmail alert: {str(e)}")

# Main processing loop
def main():
    last_alert_time = 0
    alert_cooldown = 5
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break

        # Validate frame
        if frame is None or frame.size == 0:
            print(f"Warning: Invalid frame at frame_count={frame_count}, skipping")
            continue
        if frame.shape[2] != 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLOv11 inference
        try:
            results = model(frame, conf=0.35, iou=0.1, device='cpu')
            print(f"Inference successful at frame {frame_count}, detected {len(results)} objects")
        except Exception as e:
            print(f"Error during YOLOv11 inference at frame {frame_count}: {e}")
            continue

        annotated_frame = frame.copy()
        fire_detected_yolo = False
        smoke_detected_yolo = False

        # Process detections (fire and smoke)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = model.names[class_id]
                if class_name in ['fire', 'smoke'] and confidence >= 0.35:
                    print(f"Detected: {class_name} with confidence {confidence:.2f}")
                    if class_name == 'fire':
                        fire_detected_yolo = True
                    elif class_name == 'smoke':
                        smoke_detected_yolo = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Send alerts for fire or smoke detection
        current_time = time.time()
        if (fire_detected_yolo or smoke_detected_yolo) and (current_time - last_alert_time) > alert_cooldown:
            message = f"Flare Guard Alert: {'Fire' if fire_detected_yolo else ''}{' and ' if fire_detected_yolo and smoke_detected_yolo else ''}{'Smoke' if smoke_detected_yolo else ''} detected!"
            threading.Thread(target=send_gmail_alert, args=(annotated_frame, message, LATITUDE, LONGITUDE)).start()
            last_alert_time = current_time

        # Write frame to output
        out.write(annotated_frame)
        frame_count += 1

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"Output video saved successfully to {output_path} ({frame_count} frames written)")
    else:
        print(f"Error: Output video at {output_path} was not created or is empty")
    sys.exit(0)

# Run the main loop
if __name__ == "__main__":
    main()