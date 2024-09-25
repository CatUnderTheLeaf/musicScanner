import os
from obb_anns import OBBAnns
import yaml


ds_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dataset')
cur_dir = os.path.dirname(os.path.realpath(__file__))
# o_train = OBBAnns(os.path.join(ds_dir, 'deepscores_train.json'))
# o_train.load_annotations()
# o_train.set_annotation_set_filter(['deepscores'])

# with open(os.path.join(ds_dir, 'train.txt'), 'w') as f:
#     for x in o_train.img_info:
#         f.write('./images/'+x['filename']+'\n')

o_test = OBBAnns(os.path.join(ds_dir, 'deepscores_test.json'))
o_test.load_annotations()
o_test.set_annotation_set_filter(['deepscores'])

# with open(os.path.join(ds_dir, 'test.txt'), 'w') as f:
#     for x in o_test.img_info:
#         f.write('./images/'+x['filename']+'\n')

_classes = {
    'path': ds_dir, # dataset root dir
    'train': 'train.txt', # train images (relative to 'path') 
    'test': 'test.txt', # val images (relative to 'path')
    # 'names': {k-1:v["name"] for (k, v) in o_train.get_cats().items()}
    }
# print(_classes)
print()

# yaml.dump(_classes, open(os.path.join(cur_dir, 'document.yaml'), 'w')) 
