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
# model = YOLOv10.from_pretrained('jameslahm/yolov10n').to(device)
# for other usages past here the path to the last training
model_weights = '/home/cat/projects/musicScanner/runs/detect/train/weights/best.pt'
model = YOLOv10(model_weights).to(device)


cur_dir = os.path.dirname(os.path.realpath(__file__))

results = model.train(data=os.path.join(cur_dir, 'deepscore.yaml'), epochs=100, batch=1, imgsz=1024)


# dfl_loss, it stands for "distribution focal loss", 
# which is a variant of focal loss that helps improve model 
# performance when training data is imbalanced. 
# Specifically, distribution focal loss is used to deal with class 
# imbalance that arises when training on datasets with very rare objects. 
# When there are very few examples of a certain object class in the training set, 
# the network can typically struggle to learn to detect these objects properly. 
# The distribution focal loss aims at addressing this problem and making 
# sure that the model correctly detects these rare objects.

# Box_om (Objectness Matching Box Loss): This loss measures the error in the predicted bounding box coordinates for object matches. It typically uses a combination of IoU and GIoU (Generalized IoU) losses to ensure better alignment with the ground truth.
# [
# \text{Box_om} = \lambda_{\text{box}} \sum_{i=0}^{N} \mathbb{1}_{i}^{\text{obj}} \left[ \text{IoU}(B_i, \hat{B}i) + \text{GIoU}(B_i, \hat{B}i) \right]
# ]
# where ( \lambda{\text{box}} ) is a scaling factor, ( N ) is the number of bounding boxes, and ( \mathbb{1}{i}^{\text{obj}} ) indicates if an object is present.

# Box_oo (Objectness Overlap Box Loss): This loss focuses on the overlap between predicted and ground truth boxes, using IoU and DIoU (Distance IoU) for better spatial alignment.
# [
# \text{Box_oo} = \lambda_{\text{box}} \sum_{i=0}^{N} \mathbb{1}_{i}^{\text{obj}} \left[ \text{IoU}(B_i, \hat{B}_i) + \text{DIoU}(B_i, \hat{B}_i) \right]
# ]

# Cls_om (Objectness Matching Class Loss): This measures the error in the predicted class probabilities for object matches, using Focal Loss to handle class imbalance.
# [
# \text{Cls_om} = \sum_{i=0}^{N} \mathbb{1}{i}^{\text{obj}} \sum{c \in \text{classes}} \left[ \text{FL}(p_i(c), \hat{p}_i(c)) \right]
# ]
# where ( \text{FL} ) denotes Focal Loss.

# Cls_oo (Objectness Overlap Class Loss): Similar to Cls_om but focuses on overlapping objects, ensuring better classification accuracy in dense scenes.
# [
# \text{Cls_oo} = \sum_{i=0}^{N} \mathbb{1}{i}^{\text{obj}} \sum{c \in \text{classes}} \left[ \text{FL}(p_i(c), \hat{p}_i(c)) \right]
# ]

# DFL_om (Objectness Matching Distribution Focal Loss): This loss is used for fine-grained classification, focusing on hard-to-classify examples.
# [
# \text{DFL_om} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
# ]

# DFL_oo (Objectness Overlap Distribution Focal Loss): Similar to DFL_om but applied to overlapping objects.
# [
# \text{DFL_oo} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
# ]

