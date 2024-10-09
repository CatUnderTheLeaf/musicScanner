# from ultralytics import YOLOv10
# from ultralytics import YOLOWorld
import cv2
import torch
import os

image_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dataset', 'deepscore_sliced', 'images')
# print(image_dir)

print(torch.__version__)
print(torch.version.cuda)

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_weights = '/home/cat/projects/musicScanner/runs/detect/train10/weights/best.pt'
# model = YOLOv10(model_weights).to(device)
# print(model.device.type)
# source = image_dir+'/lg-2267728-aug-beethoven--page-2.png_512_2048_1152_2688.png'
# stream = False
# save = True
# save = False


# results = model.predict(source=source, stream=stream, save=save, conf=0.5)
# for result in results:
#     det_annotated = result.plot(show=False)
#     # det_annotated[0].verbose()
#     cv2.imshow('image', det_annotated)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#       cv2.destroyAllWindows()
#       break 


from sahi.predict import get_sliced_prediction
from additional_classes.yolo10_sahi_detection_model import Yolov10DetectionModel

image_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dataset', 'deepscore', 'images')
source = image_dir+'/lg-2267728-aug-beethoven--page-2.png'
detection_model = Yolov10DetectionModel(
    model_path=model_weights,
    confidence_threshold=0.1,
    device="cuda:0", # 'cpu' or 'cuda:0'
)


result = get_sliced_prediction(
    source,
    detection_model,
    slice_height = 640,
    slice_width = 640,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)
# result = get_prediction(source, detection_model)
result.export_visuals(export_dir="results/", hide_labels=True, rect_th=2, text_size=1)