from ultralytics import YOLOv10
import os
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


cur_dir = os.path.dirname(os.path.realpath(__file__))

results = model.train(data=os.path.join(cur_dir, 'deepscore.yaml'), epochs=1, imgsz=320, fraction=0.2)
