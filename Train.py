from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(
    data=r'vestidura-trabajo-vs.v2i.yolov8/data.yaml',
    epochs=25,      # Number of epochs, i.e. cycles through the data (default: 3000) if low epochs, use high learning rate and vice versa
    imgsz=640       # Image size (pixels) 640x640
    )
