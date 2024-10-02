from sahi.slicing import slice_image, annotation_inside_slice

from sahi.utils.shapely import ShapelyAnnotation, box

import os
from pathlib import Path

ds_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dataset', 'deepscore')

output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dataset', 'deepscore_sliced')

imgsz = 640


def convert_coco_to_yolo_boxes(coco_box, w, h):
    new_center_x = (coco_box[0] + coco_box[2] / 2) / w
    new_center_y = (coco_box[1] + coco_box[3] / 2) / h
    new_width = coco_box[2] / w
    new_height = coco_box[3] / h
    return f"{new_center_x:.5f} {new_center_y:.5f} {new_width:.5f} {new_height:.5f}"

def slice_yolo_ds(input_dir, output_dir, imgsz, min_area_ratio=0.1):
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)  # create labels dir
    image_list = os.listdir(os.path.join(input_dir, 'images'))
    for image in image_list:
        # print(image)
        in_f_name = os.path.join(input_dir, 'images', image)
        out_f_name = os.path.join(output_dir, 'images', image)
        # print(out_f_name)
        l_f_name = os.path.join(input_dir, 'labels', Path(image).stem + ".txt")
        # print(l_f_name)
        result = slice_image(in_f_name, output_dir=os.path.join(output_dir, 'images'), output_file_name=out_f_name, slice_height=imgsz, slice_width=imgsz)
        with open(l_f_name, 'r') as f:
            annotations = f.readlines()
        width, height = result.original_image_width, result.original_image_height
        # print(annotations)
        ann_list = []
        for ann in annotations:
            x, y, w, h = map(float, ann.strip().split()[1:])
            x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
            x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
            id = ann.strip().split()[0]
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]            
            ann_list.append({'id': id, 'bbox': bbox})

        for im in result.sliced_image_list:
            # print(im.coco_image.file_name)
            # print(Path(im.coco_image.file_name).stem + ".txt")
            out_l_f_name = os.path.join(output_dir, 'labels', Path(im.coco_image.file_name).stem + ".txt")
            # print(out_l_f_name)
            slice = im.starting_pixel
            slice.append(slice[0]+imgsz)
            slice.append(slice[1]+imgsz)
            new_ann = []
            for ann in ann_list:
                ann_in_slice = annotation_inside_slice(ann, slice)
                # print(ann_in_slice)
                if ann_in_slice:
                    shapely_polygon = box(slice[0], slice[1], slice[2], slice[3])
                    shapely_annotation = ShapelyAnnotation.from_coco_bbox(bbox=ann['bbox'])
                    intersection_shapely_annotation = shapely_annotation.get_intersection(shapely_polygon)
                    # print(intersection_shapely_annotation)
                    new_bbox = ShapelyAnnotation.to_xywh(intersection_shapely_annotation)
                    if shapely_annotation.area:
                        if (intersection_shapely_annotation.area / shapely_annotation.area) >= min_area_ratio:
                    # # print(new_bbox)
                    # if new_bbox:
                            new_yolo_box = convert_coco_to_yolo_boxes(new_bbox, imgsz, imgsz)
                            new_ann.append(f"{ann['id']} {new_yolo_box}\n")
            if new_ann:
                # print("not empty")
                with open(out_l_f_name, 'w') as f:
                    f.writelines(new_ann)
        

slice_yolo_ds(ds_dir, output_dir, imgsz)