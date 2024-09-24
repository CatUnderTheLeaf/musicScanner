# musicScanner
Optical Music Recognition using Deep Learning

### Dataset - DeepScoresV2[^1]
The DeepScoresV2 Dataset for Music Object Detection contains digitally rendered images of written sheet music, together with the corresponding ground truth to fit various types of machine learning models. A total of 151 Million different instances of music symbols, belonging to 135 different classes are annotated. The total Dataset contains 255,385 Images. For most researches, the dense version, containing 1714 of the most diverse and interesting images, should suffice.

The dataset contains ground in the form of:
- Non-oriented bounding boxes
- Oriented bounding boxes
- Semantic segmentation
- Instance segmentation

Download [here](https://zenodo.org/records/4012193).

The accompaning paper [The DeepScoresV2 Dataset and Benchmark for Music Object Detection published at ICPR2020](https://digitalcollection.zhaw.ch/items/e40b1bb1-821e-4504-8df0-e6f72a650210).

Other [datasets](https://apacha.github.io/OMR-Datasets/).

### obb_anns
A toolkit for convenient loading and inspection of the data was [copied](https://github.com/yvan674/obb_anns) and I have made changes, because some of the external libraries have deprecated functions.


[^1]: L. Tuggener, Y. P. Satyawan, A. Pacha, J. Schmidhuberand T. Stadelmann, “DeepScoresV2”. Zenodo, Sep. 02, 2020. doi: 10.5281/zenodo.4012193.
