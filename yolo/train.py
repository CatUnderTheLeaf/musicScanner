from ultralytics import YOLOv10
# from ultralytics import YOLOWorld
import torch

print(torch.__version__)
print(torch.version.cuda)

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Load a pretrained YOLOv8s-worldv2 model
# for the first usage:
# model_last_weights = "yolov8s-worldv2.pt"
# for other usages past here the path to the last training
model = YOLOv10.from_pretrained('jameslahm/yolov10s').to(device)

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="lvis.yaml", epochs=4, imgsz=224, fraction=0.2)
