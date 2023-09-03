from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('model/best.pt')

# Run inference on data/1_min_cam_4.mp4 with arguments
model.predict('data/1_min_cam_4.mp4', save=True, imgsz=320, conf=0.5)