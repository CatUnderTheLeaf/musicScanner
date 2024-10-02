# from obb_anns import OBBAnns
# o = OBBAnns('dataset/deepscores_test.json')
# o.load_annotations()
# o.set_annotation_set_filter(['deepscores'])

# # Get the first 50 images
# # img_idxs = [i for i in range(50)]
# # imgs, anns = o.get_img_ann_pairs(idxs=img_idxs)

# # Visualize immediately
# # o.visualize(img_idx=1, out_dir='results', show=False, instances=True)

# # Get the first 5 images
# img_idxs = [i for i in range(1)]
# imgs, anns = o.get_img_ann_pair(idxs=img_idxs)
# print(anns[0])
# # print(anns[0].loc[160133])
# _classes = [v["name"] for (k, v) in o.get_cats().items()]
# print(_classes)

import supervision as sv
import os
from PIL import Image

dataset = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dataset', 'deepscore_sliced')
cur_dir = os.path.dirname(os.path.realpath(__file__))

ds = sv.DetectionDataset.from_yolo(
    images_directory_path=f'{dataset}/images',
    annotations_directory_path=f'{dataset}/labels',
    data_yaml_path=os.path.join(cur_dir, 'deepscore_sliced.yaml')
)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_images = []
os.makedirs(os.path.join(cur_dir, 'test'), exist_ok=True)  # create labels dir
for i in range(16):
    _, image, annotations = ds[i]

    labels = [ds.classes[class_id] for class_id in annotations.class_id]

    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(annotated_image, annotations)
    annotated_image = label_annotator.annotate(annotated_image, annotations, labels)
    annotated_images.append(annotated_image)
    Image.fromarray(annotated_image).save(os.path.join(cur_dir, 'test', f"{i}_deep.png"))
    # print(annotations)
