import streamlit as st
import numpy as np

import cv2

import scanner.detection as detector

st.title('Scan your music')

def get_photo(img_file, key):
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    notes_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    det_im = detector.detect_everything(notes_image, hide_labels=False, rect_th=2, text_size=1)
    st.image(det_im)

def main():
    image = st.file_uploader(label_visibility = "collapsed", type=['png', 'jpg'], label='upload picture')
    if image:
        get_photo(image, 'upload')


if __name__ == "__main__":
    main()