import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

#list all csv files only
csv_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'runs', 'detect', 'train')
csv_files = sorted(glob.glob(csv_dir+ '*/*.{}'.format('csv')))
# print(csv_files)

df_concat = pd.concat([pd.read_csv(f) for f in csv_files ], ignore_index=True)
print(df_concat.columns)

df_concat[['           train/box_om', '             val/box_om']].plot()
plt.savefig('results/box_om.png')

df_concat[['           train/cls_om', '             val/cls_om']].plot()
plt.savefig('results/cls_om.png')

df_concat[['           train/dfl_om', '             val/dfl_om']].plot()
plt.savefig('results/dfl_om.png')

df_concat[['           train/box_oo', '             val/box_oo']].plot()
plt.savefig('results/box_oo.png')

df_concat[['           train/cls_oo', '             val/cls_oo']].plot()
plt.savefig('results/cls_oo.png')

df_concat[['           train/dfl_oo', '             val/dfl_oo']].plot()
plt.savefig('results/dfl_oo.png')

df_concat[['   metrics/precision(B)', '      metrics/recall(B)', '       metrics/mAP50(B)', '    metrics/mAP50-95(B)']].plot()
plt.savefig('results/metrics.png')