from ultralytics import YOLOv10
# from ultralytics import YOLOWorld
import cv2
import torch
import os

image_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dataset', 'images')
# print(image_dir)

print(torch.__version__)
print(torch.version.cuda)

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLOv10.from_pretrained('jameslahm/yolov10x').to(device)
print(model.device.type)

source = image_dir+'/lg-3948783-aug-gonville--page-1.png'
stream = False
save = True
# save = False


results = model.predict(source=source, stream=stream, save=save, conf=0.01)
for result in results:
    det_annotated = result.plot(show=False)
    # det_annotated[0].verbose()
    cv2.imshow('image', det_annotated)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break 
