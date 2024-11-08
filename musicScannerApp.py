import streamlit as st

import scanner.detection as detector
import scanner.utils as utils

# make a wide layout, not with a fixed width in the center 
st.set_page_config(
    page_title="Music sheet scanner",
    layout="wide",
)

st.title('Scan your music')

def main():
    uploaded_image = st.file_uploader(label_visibility = "collapsed", type=['png', 'jpg', 'jpeg','pdf'], label='upload picture') #, accept_multiple_files=True
    
    decoded_image = utils.load_test_image()
    caption = 'test_image'
    if uploaded_image:
        decoded_image = utils.get_photo(uploaded_image, 'upload')
        caption = ''
    
    detections = detector.detect_everything(decoded_image)
    classes_to_view = set(['staff'])

    sliced_staffs = detector.slice_image(detections, divider='staff')
    for staff in sliced_staffs:
        visual_staff = detector.visualize_predictions(staff['image'], staff['predictions'], hide_labels=False, rect_th=2, text_size=1) # , filter=classes_to_view
        st.image(visual_staff, caption)
        st.divider()
        

if __name__ == "__main__":
    main()