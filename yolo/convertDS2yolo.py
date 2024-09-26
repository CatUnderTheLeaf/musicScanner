import os
from obb_anns import OBBAnns
import yaml


ds_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dataset', 'deepscore')

o = OBBAnns(os.path.join(ds_dir, 'deepscores_train.json'))
o.load_annotations()
o.set_annotation_set_filter(['deepscores'])
_classes = {
    'path': ds_dir, # dataset root dir
    'train': 'deepscores_train.txt', # train images (relative to 'path') 
    'val': 'deepscores_val.txt', # val images (relative to 'path')
    'test': 'deepscores_test.txt', # test images (relative to 'path')
    'names': {k-1:v["name"] for (k, v) in o.get_cats().items()}
    }

cur_dir = os.path.dirname(os.path.realpath(__file__))
yaml.dump(_classes, open(os.path.join(cur_dir, 'deepscore.yaml'), 'w')) 

# just copy this to the yaml file
data = """
download: |
  import shutil
  import os
  from pathlib import Path
  from obb_anns import OBBAnns

  from ultralytics.utils.downloads import download
  from ultralytics.utils.ops import xyxy2xywh
  
  # Download
  dir = Path(yaml['path'])  # dataset root dir
  if not dir.exists():
      parent = Path(dir.parent)  # download dir
      urls = ['https://zenodo.org/records/4012193/files/ds2_dense.tar.gz?download=1']
      download(urls, dir=parent)
      # Rename directories
      if dir.exists():
          shutil.rmtree(dir)
      (parent / 'ds2_dense').rename(dir)  # rename dir
      os.remove(parent / 'ds2_dense.tar.gz')

  (dir / 'labels').mkdir(parents=True, exist_ok=True)  # create labels dir

  # Convert
  def convert_labels(json_file_name, split=False):
      o = OBBAnns(os.path.join(ds_dir, json_file_name))
      o.load_annotations()
      o.set_annotation_set_filter(['deepscores'])

      img_idxs = [i for i in range(len(o.img_info))]
      imgs, anns = o.get_img_ann_pair(idxs=img_idxs)

      def convert_boxes(box, w, h):
          xywh = xyxy2xywh(np.array([[box[0] / w, box[1] / h, box[2] / w, box[3] / h]]))[0]
          return f"{xywh[0]:.5f} {xywh[1]:.5f} {xywh[2]:.5f} {xywh[3]:.5f}"

      for img, ann in zip(imgs, anns):

          label_name = os.path.join(dir, 'labels', Path(img['filename']).stem + ".txt")

          w = img['width']
          h = img['height']
          # DO NOT FORGET to make -1 on cat_id
          # - The bounding boxes are given as absolute values.
          # - `a_bbox` contains the coordinates of the top left and bottom right corners.
          # - `cat_id` is stringified ints and starts at 1.
          result = [f"{cat[0]-1} {convert_boxes(box, w, h)}\n" for cat, box in zip(ann['cat_id'], ann['a_bbox'])]

          with open(label_name, 'w') as f:
              f.writelines(result)
      if split:
          with open(os.path.join(ds_dir, Path(json_file_name).stem + ".txt"), 'w') as f:
              for x in imgs[:int(len(o.img_info)*0.8)]:
                  f.write('./images/'+x['filename']+'\n')
          with open(os.path.join(ds_dir, "deepscores_val.txt"), 'w') as f:
              for x in imgs[:int(len(o.img_info)*0.8)]:
                  f.write('./images/'+x['filename']+'\n')
      else:
          with open(os.path.join(ds_dir, Path(json_file_name).stem + ".txt"), 'w') as f:
              for x in imgs:
                  f.write('./images/'+x['filename']+'\n')

  convert_labels('deepscores_test.json')
  convert_labels('deepscores_train.json', split=True)
"""



# def str_presenter(dumper, data):
#     """configures yaml for dumping multiline strings
#     Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data"""
#     if data.count('\n') > 0:  # check for multiline string
#         return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
#     return dumper.represent_scalar('tag:yaml.org,2002:str', data)

# yaml.add_representer(str, str_presenter)
# d = yaml.load(data, yaml.Loader)
# yaml.dump(d, open(os.path.join(cur_dir, 'document.yaml'), 'a'))
