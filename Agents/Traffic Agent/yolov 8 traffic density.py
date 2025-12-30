
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 21:33:01 2025
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
GMAIL_APP_PASSWORD = "Password"  # Replace with your Gmail App Password
RECIPIENT_EMAIL = "usama.aba18@gmail.com"

# Initial latitude and longitude for Karachi
LATITUDE = 24.8607
LONGITUDE = 67.0011

# Model and video paths
model_path = r"C:\Users\FMT COMPUTERS\Downloads\Code Paper 3\sPYDER\YOLOv8_Traffic_Density_Estimation\models\best.pt"
video_path = r"C:\Users\FMT COMPUTERS\Downloads\Code Paper 3\sPYDER\YOLOv8_Traffic_Density_Estimation\sample_video.mp4"  # Repository's sample video
output_dir = r"C:\Users\FMT COMPUTERS\Downloads\Code Paper 3\sPYDER\YOLOv8_Traffic_Density_Estimation"
output_path = os.path.join(output_dir, "output4.mp4")
alerts_json_path = os.path.join(output_dir, "traffic_alerts.json")
debug_frames_dir = os.path.join(output_dir, "debug_frames")

# Traffic density threshold
DENSITY_THRESHOLD = 20  # Adjusted for realistic high density

# Validate paths
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit(1)
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    sys.exit(1)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(debug_frames_dir):
    os.makedirs(debug_frames_dir)

# Initialize alerts JSON if not exists
if not os.path.exists(alerts_json_path):
    with open(alerts_json_path, 'w') as f:
        json.dump([], f)

# Load YOLOv8 model
try:
    model = YOLO(model_path)
    print(f"Model classes: {model.names}")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}, trying pretrained yolov8n.pt")
    try:
        model = YOLO("yolov8n.pt")  # Fallback to pretrained COCO model
        print(f"Fallback model classes: {model.names}")
    except Exception as e:
        print(f"Error loading fallback model: {e}")
        sys.exit(1)

# Verify expected vehicle class
vehicle_classes = ['Vehicle']  # As per repository's dataset
if 'Vehicle' not in model.names.values():
    print(f"Warning: 'Vehicle' class not found in model classes {list(model.names.values())}")
    print("Switching to COCO classes: ['car', 'truck', 'bus', 'motorcycle']")
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
    if not any(cls in model.names.values() for cls in vehicle_classes):
        print(f"Error: No vehicle classes {vehicle_classes} found in model. Exiting.")
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

# Resize to match model’s training resolution (640x640)
target_size = (640, 640)

# Initialize video writer with XVID codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
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
def send_gmail_alert(image, message, lat, long, vehicle_count):
    if GMAIL_APP_PASSWORD == "your_app_password":
        print("Error: Gmail App Password not configured. Skipping email alert.")
        return
    try:
        enhanced_message = f"{message}\nLocation: Latitude {lat}, Longitude {long}\nVehicle Count: {vehicle_count}"
        temp_image_path = os.path.join(output_dir, "temp_traffic_alert_image.jpg")
        if not cv2.imwrite(temp_image_path, image):
            raise Exception("Failed to save temporary image")
        
        msg = MIMEMultipart()
        msg['From'] = GMAIL_ADDRESS
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = "Traffic Alert: High Traffic Density Detected"
        msg.attach(MIMEText(enhanced_message, 'plain'))
        with open(temp_image_path, 'rb') as img_file:
            img = MIMEImage(img_file.read(), name="traffic_image.jpg")
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
            "longitude": long,
            "vehicle_count": vehicle_count
        }
        threading.Thread(target=log_alert_to_json, args=(alert_details,)).start()
    except Exception as e:
        print(f"Error sending Gmail alert: {str(e)}")

# Main processing loop
def main():
    last_alert_time = 0
    alert_cooldown = 30  # Seconds between alerts
    frame_count = 0

    # Open window for real-time visualization
    cv2.namedWindow("Traffic Detection", cv2.WINDOW_NORMAL)

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

        # Resize frame to 640x640 (model’s training resolution)
        frame = cv2.resize(frame, target_size)

        # YOLOv8 inference
        try:
            results = model(frame, conf=0.25, iou=0.45, device='cpu')  # Lowered thresholds
            print(f"Inference successful at frame {frame_count}, detected {len(results[0].boxes)} objects")
        except Exception as e:
            print(f"Error during YOLOv8 inference at frame {frame_count}: {e}")
            continue

        annotated_frame = frame.copy()
        vehicle_count = 0

        # Process detections
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = model.names[class_id]
                if class_name in vehicle_classes and confidence >= 0.25:
                    vehicle_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    print(f"Detected: {class_name} with confidence {confidence:.2f} at ({x1}, {y1}, {x2}, {y2})")

        # Log vehicle count for debugging
        print(f"Frame {frame_count}: Detected {vehicle_count} vehicles")

        # Add vehicle count to frame
        cv2.putText(annotated_frame, f"Vehicle Count: {vehicle_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Send alert for high traffic density
        current_time = time.time()
        if vehicle_count >= DENSITY_THRESHOLD and (current_time - last_alert_time) > alert_cooldown:
            message = f"Traffic Alert: High traffic density detected! {vehicle_count} vehicles observed."
            threading.Thread(target=send_gmail_alert, args=(annotated_frame, message, LATITUDE, LONGITUDE, vehicle_count)).start()
            last_alert_time = current_time

        # Save debug frame
        debug_frame_path = os.path.join(debug_frames_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(debug_frame_path, annotated_frame)

        # Display frame
        cv2.imshow("Traffic Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
