from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
from .yolo10_sahi_detection_model import Yolov10DetectionModel
import torch
import numpy as np
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'weights/best.pt')

detection_model = Yolov10DetectionModel(
    model_path=model_weights,
    confidence_threshold=0.5,
    device="cuda:0", # 'cpu' or 'cuda:0'
)

def detect_everything(
        source: np.array, 
        text_size: float = None,
        rect_th: int = None,
        hide_labels: bool = False,
        hide_conf: bool = False):
    
    result = get_sliced_prediction(
        source,
        detection_model,
        slice_height = 640,
        slice_width = 640,
        overlap_height_ratio = 0.2,
        overlap_width_ratio = 0.3
    )

    im_with_det = visualize_object_predictions(
            image=np.ascontiguousarray(result.image),
            object_prediction_list=result.object_prediction_list,
            rect_th=rect_th,
            text_size=text_size,
            text_th=None,
            color=None,
            hide_labels=hide_labels,
            hide_conf=hide_conf
        )
    
    return im_with_det["image"]