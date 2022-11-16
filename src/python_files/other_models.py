import streamlit as st
from skimage import io
from PIL import Image
from io import BytesIO
import os
import numpy as np
import cv2 as cv

import streamlit.components.v1 as components
import time
import tensorflow as tf

import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt

# For drawing onto the image.
import numpy as np
import pandas as pd
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time, random
import easyocr

module_handle1 = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
reader = easyocr.Reader(['en'])

def display_image(image):
    st.image(image)

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=20, display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getbbox(ds)[3] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getbbox(display_str)[2:4]
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin
    return [left, right, top, bottom]


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())
    detections = []
    det_scores = []

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
    except IOError:
        #print("Font not found, using default font.")
        #font = ImageFont.load_default()
        font = ImageFont.truetype("arial.ttf", 15)


    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                         int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            det_scores = scores[i]
            detections=draw_bounding_box_on_image(
              image_pil,
              ymin,
              xmin,
              ymax,
              xmax,
              color,
              font,
              display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image, detections, det_scores

def load_img(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

def run_detector(detector, img):
    img = load_img(img)
    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {key:value.numpy() for key,value in result.items()}

    duration = "Time used for detecting " + str(round(end_time-start_time, 1))

    image_with_boxes = draw_boxes(
        img,
        result["detection_boxes"],
        result["detection_class_entities"],
        result["detection_scores"])

    # display_image(image_with_boxes)
    return result,image_with_boxes,duration

def detecting(image,classifier,detector,min_score=0.1):
    returned = []
    result,img_box,duration = run_detector(detector, image)
    detecting = np.where((result['detection_class_entities']==classifier) & (result['detection_scores'] > min_score))[0]
    found_objects = "Found %d objects." % len(detecting)
    if len(detecting) == 0:
        return "Not Found", result, found_objects, duration
    return detecting, result, found_objects, duration

def read_files():
    data = os.listdir('../data/raw/')
    data.sort()
    snaps = os.listdir('../data/snaps/')
    snaps.sort()
    return data,snaps

def st_other_models():
    global filenames
    data,snaps=read_files()
    if st.checkbox("Upload Images"):
        select_image = st.selectbox("Choose a picture",options = snaps,index = 0, key = 1)
        path = '../data/snaps/'
    else:
        select_image = st.selectbox("Choose a picture",options = data,index = 1, key = 1)
        path = '../data/raw/'
    select_detecting = st.selectbox("Choose a detected object",options = ['Person','Car','Truck','Bus','Vehicle registration plate'],index = 1, key=101)
    select_detector = st.radio("Select Model", ("MobileNet",), key=102)
    select_min_score = st.slider('Min Score of Detecting %',key=104, min_value = 10, max_value = 100, value=20, step = 1)
    st.header("Example of detected image")
    uploaded_file = st.file_uploader("Choose a file for detecting image",["jpg","png"], key=103)
    col1, col2 = st.columns(2)
    col1.header("Image")
    if isinstance(uploaded_file,BytesIO):
        if snaps:
            count = int(snaps[-1].split(".")[0].split('_')[1])
            if not uploaded_file.name in filenames:
                count += 1
        else:
            count = 0
        path_write = "../data/snaps/car_"+(5-len(str(count)))*"0"+str(count)+".jpg"
        img = Image.open(uploaded_file)
        img = np.array(img)
        if not uploaded_file.name in filenames:
            filenames.append(uploaded_file.name)
            cv.imwrite(path_write,img)
            st.experimental_rerun()
        col1.image(uploaded_file, use_column_width = True)
        det_img = path_write
    elif path == '../data/snaps/' and not snaps:
        filenames = []
        st.header("You dont upload images yet. Please, click and upload image.")
    else:
        filenames = []
        img = cv.imread(path+select_image)
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        col1.image(img, use_column_width = True)
        det_img = path+select_image
    if not (path == '../data/snaps/' and not snaps) and st.button("Run Detector"):
        with st.spinner('Loading model ...'):
            if select_detector == "MobileNet":
                if 'detector1' not in st.session_state:
                    st.session_state.detector1 = hub.load(module_handle1).signatures['default']
                detector1 = st.session_state.detector1
                cur_detector = detector1
        with st.spinner('Detecting ...'):
            if select_detector == "MobileNet":
                ret, result, found_objects, duration = detecting(det_img,select_detecting.encode(),cur_detector,min_score=select_min_score/100)
                st.write(duration)
                if ret != "Not Found":
                    st.write(found_objects)
                    det_images = []
                    det_scores = []
                    col2.header("Detected Image")
                    img = load_img(det_img)
                    real_img = img.copy()
                    for ip in ret:
                        img, det, score = draw_boxes(
                              img, result["detection_boxes"][[ip]],
                              result["detection_class_entities"][[ip]], result["detection_scores"][[ip]],max_boxes=100)
                        det = np.array(det, np.int32)
                        det_images.append(real_img[det[2]:det[3],det[0]:det[1]])
                        det_scores.append(score)
                    col2.image(img/255, use_column_width = True)
                    st.header("Detections")
                    col_car1, col_car2 = st.columns(2)
                    if(select_detecting == "Vehicle registration plate"):
                        st.header("Detected Text")
                        col_tx1, col_tx2 = st.columns(2)
                    for i in range(len(det_images)):
                        cur_det = det_images[i]
                        platetext = reader.readtext(cur_det)
                        cur_score = det_scores[i]
                        if i % 2 == 0:
                            col_car1.write(select_detecting+" by "+str(round(cur_score*100))+"% ↓")
                            col_car1.image(cur_det)
                            if(select_detecting == "Vehicle registration plate"):
                                for ind, text in enumerate(platetext):
                                    col_tx1.write(str(ind+1) + ". " + text[1])
                        else:
                            col_car2.write(select_detecting+" by "+str(round(cur_score*100))+"% ↓")
                            col_car2.image(cur_det)
                            if(select_detecting == "Vehicle registration plate"):
                                for ind, text in enumerate(platetext):
                                    col_tx1.write(str(ind+1) + ". " + text[1])
                else:
                    col2.header(ret + " " + select_detecting.split(" ")[-1])
                    col2.image(img, use_column_width = True)
