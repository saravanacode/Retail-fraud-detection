import cv2
import numpy as np
import requests
from ultralytics import YOLO

import time

class FraudDetectionSystem:
    def __init__(self, detection_model_path, classification_model_path, api_url):
        self.detector = YOLO(detection_model_path)
        self.classifier = YOLO(classification_model_path)
        self.crew_roi = (471, 154, 636, 674)  # (x, y, width, height)
        self.customer_roi = (169, 284, 388, 507)  # Example ROI, adjust as needed
        self.display_customer_count = 0
        self.display_total_invoice_count = 0
        self.display_invoice_flag = "Not Fraud"
        self.customer_variable = []
        self.api_url = api_url
    
        self.last_transaction_time = 0


    def get_crew_area(self, frame):
        x, y, w, h = self.crew_roi
        return frame[y:y+h, x:x+w]

    def is_customer_in_roi(self, customer_box):
        x1, y1, x2, y2 = customer_box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rx, ry, rw, rh = self.customer_roi
        return rx < cx < rx + rw and ry < cy < ry + rh

    def detect_fraud(self, video_path):
        cap = cv2.VideoCapture(video_path)
        customer_data = {}
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create VideoWriter object
        output_path = 'output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
  

            # Detect and track objects in the frame
            results = self.detector.track(frame, persist=True, conf=0.6, iou=0.3, verbose=False,tracker="bytetrack.yaml")

            primary_customer = None
            max_area = 0

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()

                for box, track_id in zip(boxes, track_ids):
                    if self.is_customer_in_roi(box):
                        x1, y1, x2, y2 = map(int, box)
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            primary_customer = (track_id, box)

            if primary_customer:
                track_id, box = primary_customer
                x1, y1, x2, y2 = map(int, box)

                if track_id not in customer_data:
                    customer_data[track_id] = {'invoice_count': 0, 'frame_count': 0}
                
                customer_data[track_id]['frame_count'] += 1

                crew_area = self.get_crew_area(frame)
                crew_results = self.classifier(crew_area, verbose=False)[0]
                if crew_results.probs is not None and len(crew_results.probs) > 1:
                    invoice_processed_prob = crew_results.probs.data[1].item()
                    if invoice_processed_prob <= 0.5:
                        customer_data[track_id]['invoice_count'] += 1

                # Draw bounding box for primary customer
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display invoice count
                # inv_count = customer_data[track_id]['invoice_count']
                # cv2.putText(frame, f"Inv: {inv_count}", (x1, y2 + 20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check for customers who have been absent for more than 50 frames
            for track_id in list(customer_data.keys()):
                if primary_customer is None or track_id != primary_customer[0]:
                    customer_data[track_id]['frame_count'] += 1
                    if customer_data[track_id]['frame_count'] > 100:                        
                        # Update counters
                        invoice_flag = customer_data[track_id]['invoice_count'] >= 15
                        # Save result
                        self.save_result(int(track_id), invoice_flag)
                        # Update display variables
                        if track_id not in self.customer_variable:
                            self.customer_variable.append(track_id)
                            self.display_customer_count += 1
                            self.display_invoice_flag = "Not Fraud " if invoice_flag else "Fraud"
                            if str(self.display_invoice_flag) == "Not Fraud ":
                                self.display_total_invoice_count += 1
                        del customer_data[track_id]

            # Display counters and flag
            cv2.putText(frame, f"Customers: {self.display_customer_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Total Invoices: {self.display_total_invoice_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.putText(frame, f"Invoice Flag: customer Id :{self.customer_variable[-1] if self.customer_variable else 'None'} {int(self.display_invoice_flag)}", (10, 90),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            invoice_flag_text = str(self.display_invoice_flag)
            cv2.putText(frame, f"Invoice Flag: customer Id :{self.customer_variable[-1] if self.customer_variable else 'None'} {invoice_flag_text}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
            # print(customer_data)
            frame = cv2.resize(frame, (1280, 720))
            # Write the frame to the output video
            out.write(frame)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video saved as {output_path}")

    def save_result(self, customer_id, invoice_provided):
        data = {
            'customer_id': customer_id,
            'invoice_provided': int(invoice_provided)
        }
        try:
            response = requests.post(f"{self.api_url}/save_result", json=data)
            if response.status_code == 200:
                print(f"Result saved for customer {customer_id}")
            else:
                print(f"Failed to save result for customer {customer_id}")
        except requests.RequestException as e:
            print(f"Error saving result: {e}")

if __name__ == "__main__":
    detection_model_path = "yolov8n.pt"  # Path to your object detection model
    classification_model_path = "fraud_detection_model.pt"  # Path to your classification model
    api_url = "http://localhost:5000"  # Update this with your actual API URL

    fraud_detector = FraudDetectionSystem(detection_model_path, classification_model_path, api_url)
    test_video = "multiple_cash_transaction_raw_example.mp4"
    # ztest_video = "cash_no_invoice_30sec.mp4"

    fraud_detector.detect_fraud(test_video)