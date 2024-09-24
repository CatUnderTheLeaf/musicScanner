from obb_anns import OBBAnns
o = OBBAnns('dataset/deepscores_test.json')
o.load_annotations()
o.set_annotation_set_filter(['deepscores'])

# Get the first 50 images
# img_idxs = [i for i in range(50)]
# imgs, anns = o.get_img_ann_pairs(idxs=img_idxs)

# Visualize immediately
# o.visualize(img_idx=1, out_dir='results', show=False, instances=True)

# Get the first 5 images
img_idxs = [i for i in range(5)]
imgs, anns = o.get_img_ann_pair(idxs=img_idxs)
# print(anns[0])
_classes = [v["name"] for (k, v) in o.get_cats().items()]
print(_classes)