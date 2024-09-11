import cv2
import numpy as np
from ultralytics import YOLO
import os


class FraudDetectionSystem:
    def __init__(self):
        self.classifier = YOLO('yolov8n-cls.pt')  # Load a pretrained YOLOv8 classification model
        self.is_trained = False
        # Define ROIs for customer and crew member areas
        # These should be adjusted based on your specific camera setup
        self.crew_roi = (471, 154, 636, 674) # (x, y, width, height)

    def get_crew_area(self, frame):
        x, y, w, h = self.crew_roi
        return frame[y:y+h, x:x+w]

    def train(self, video_paths, labels):
        base_dir = os.path.abspath("fraud_detection_data")
        dataset_dir = os.path.join(base_dir, "dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create subdirectories for each class
        class_names = ['no_invoice', 'invoice_processed']
        for class_name in class_names:
            os.makedirs(os.path.join(dataset_dir, 'train', class_name), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, 'val', class_name), exist_ok=True)
        
        dataset = []
        for video_path, label in zip(video_paths, labels):
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % 10 == 0:  # Sample every 10th frame
                    crew_area = self.get_crew_area(frame)
                    class_name = class_names[label]
                    if frame_count % 50 == 0:  # Every 5th sampled frame goes to validation
                        subset = 'val'
                    else:
                        subset = 'train'
                    crop_path = os.path.join(dataset_dir, subset, class_name, f"frame_{len(dataset)}.jpg")
                    cv2.imwrite(crop_path, crew_area)
                    dataset.append((crop_path, label))
                frame_count += 1
            cap.release()

        # Train the classification model
        self.classifier.train(data="fraud_detection_data/dataset", epochs=10, imgsz=224)
        self.is_trained = True

        # Evaluate the model
        results = self.classifier.val()
        print(f"Model performance:")
        print(f"Top-1 Accuracy: {results.top1:.4f}")
        print(f"Top-5 Accuracy: {results.top5:.4f}")
        print(f"Fitness: {results.fitness:.4f}")

# Usage example
if __name__ == "__main__":
    fraud_detector = FraudDetectionSystem()
    
    # Train the model (you'll need to provide actual video paths and labels)
    train_videos = ["./cash_no_invoice_30sec.mp4", "./multiple_cash_transaction_raw_example.mp4"]
    train_labels = [0, 1]  # 0 for no invoice processed, 1 for invoice processed
    fraud_detector.train(train_videos, train_labels)
    # Save the trained model
    model_save_path = "fraud_detection_model.pt"
    fraud_detector.classifier.save(model_save_path)
    
    print(f"Model saved to {model_save_path}")

    
