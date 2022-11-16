import streamlit as st
from skimage import io
from io import BytesIO
import os
import cv2 as cv
import streamlit.components.v1 as components
import time
import tensorflow as tf
import easyocr
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageColor, ImageDraw, ImageFont, ImageOps, Image
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from scipy.ndimage import interpolation as inter
import time
import glob
import pytesseract as pt

def img_resize(im, size):
    desired_size = size
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = desired_size/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = cv.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT,value=color)
    return new_im, top, left


interpreter = tf.lite.Interpreter(model_path='../data/models/model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
category_index = label_map_util.create_category_index_from_labelmap('../data/models/label_map.pbtxt')


reader = easyocr.Reader(['en'])

def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv.warpAffine(image, M, (w, h), flags=cv.INTER_CUBIC, \
              borderMode=cv.BORDER_REPLICATE)
    return best_angle, rotated

# @st.cache
def read_files():
#     data = os.listdir('../../data/Label_image_800/')
    data = os.listdir('../data/raw/')
    data.sort()
    snaps = os.listdir('../data/snaps/')
    snaps.sort()
    return data,snaps

def st_our_model_video_tflite():
    select_min_score = st.slider('Min Score of Detecting %',min_value = 10, max_value = 100, value=20, step = 1)
    if st.button("Run Detector"):
        with st.spinner('Detecting ...'):
            start_t = time.time()
            cap = cv.VideoCapture('test_video.mp4')
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            out = cv.VideoWriter('detected_plate_tflite.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
            if (cap.isOpened()== False):
                print("Error opening video stream or file")
            while(cap.isOpened()):
                ret, real_img = cap.read()
                if ret == True:
                    img = Image.fromarray(real_img.copy())
                    img = img.resize((320, 320), Image.ANTIALIAS)
                    real_img = np.array(real_img)
                    img = np.array(img)
                    top = 0
                    left = 0
                    image_np = img.copy()
                    input_tensor = np.array(np.expand_dims(img,0), dtype=np.float32)
                    input_index = interpreter.get_input_details()[0]["index"]
                    interpreter.allocate_tensors()
                    interpreter.set_tensor(input_index, input_tensor)
                    interpreter.invoke()
                    output_details1 = interpreter.get_output_details()[0]
                    output_details2 = interpreter.get_output_details()[1]
                    output_details3 = interpreter.get_output_details()[3]
                    detection_scores = np.squeeze(interpreter.get_tensor(output_details1['index']))
                    detection_boxes = np.squeeze(interpreter.get_tensor(output_details2['index']))
                    detection_classes = np.squeeze(interpreter.get_tensor(output_details3['index'])).astype(np.int64)

                    res_det = np.where(detection_scores>select_min_score/100)

                    label_id_offset = 1
                    image = img.copy()
                    imup = []
                    plate_scores = []
                    ind = 0
                    (im_height,im_width,ch) = real_img.shape
                    detection_boxes[:,1] += left/im_width
                    detection_boxes[:,3] += left*2/im_width
                    detection_boxes[:,0] += top/im_height
                    detection_boxes[:,2] += top*2/im_height
                    ymin, xmin, ymax, xmax= detection_boxes[0]
                    bbox = np.array([xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height]).astype(int)
                    max_center = bbox[1]-bbox[0] + bbox[3]-bbox[2]
                    max_score = 0
                    for i in res_det[0]:
                        ymin, xmin, ymax, xmax = detection_boxes[i]
                        bbox = np.array([xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height]).astype(int)
                        center = bbox[1]-bbox[0] + bbox[3]-bbox[2]
                        if detection_scores[i] > max_score:
                            max_score = detection_scores[i]
                            max_center = center
                        if np.linalg.norm(max_center - center) != 0 and np.linalg.norm(max_center - center) < 120:
                            detection_boxes = np.delete(detection_boxes, i, 0)
                            detection_classes = np.delete(detection_classes, i, 0)
                            detection_scores = np.delete(detection_scores, i, 0)
                            continue
                        plate = real_img[bbox[2]:bbox[3],bbox[0]:bbox[1]].copy()
                        best_angle,plate=correct_skew(plate, delta=1, limit=25)
                        imup.append(plate)
                        plate_scores.append(detection_scores[i])
                    if len(imup) > 0:
                        viz_utils.visualize_boxes_and_labels_on_image_array(
                            real_img,
                            detection_boxes,
                            detection_classes+label_id_offset,
                            detection_scores,
                            category_index,
                            use_normalized_coordinates=True,
                            max_boxes_to_draw=5,
                            line_thickness=10,
                            min_score_thresh=(select_min_score/100),
                            agnostic_mode=False)
                    out.write(real_img)
                    if cv.waitKey(25) & 0xFF == ord('q'):
                        break
                else:
                    break

            cap.release()
            cv.destroyAllWindows()
            end_t = time.time()
            st.markdown(f"Time used for detecting {round(end_t-start_t, 1)} s")

def st_our_model_tflite():
    data,snaps=read_files()
    if st.checkbox("snaps"):
        select_image = st.selectbox("Choose a picture",options = snaps,index = 0, key = 1)
        path = '../data/snaps/'
    else:
        select_image = st.selectbox("Choose a picture",options = data,index = 0, key = 1)
        path = '../data/raw/'
    select_min_score = st.slider('Min Score of Detecting %',min_value = 10, max_value = 100, value=20, step = 1)
    st.header("Example of detected image")
    uploaded_file = st.file_uploader("Choose a file for detecting image",["jpg","png"])
    col1, col2 = st.columns(2)
    col1.header("Image")
    if isinstance(uploaded_file,BytesIO):
        if snaps:
            st.write(snaps)
            count = int(snaps[-1].split(".")[0].split('_')[1])+1
        else:
            count = int(data[-1].split(".")[0].split('_')[1])+1
        img = Image.open(uploaded_file)
        img = cv.cvtColor(np.array(img),cv.COLOR_BGR2RGB)
        path_write = "../../data/snaps/car_"+(5-len(str(count)))*"0"+str(count)+".jpg"
        col1.image(uploaded_file)
        cv.imwrite(path_write,img)
        det_img = path_write
    else:
        img = cv.imread(path+select_image)
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        col1.image(img,use_column_width=True)
        det_img = path+select_image
    if st.button("Run Detector"):
        with st.spinner('Detecting ...'):
            start_t = time.time()
            IMAGE_PATH=det_img
            real_img = Image.open(IMAGE_PATH)
            img = real_img.copy()
            img = cv.resize(np.array(img), (320, 320), interpolation = cv.INTER_NEAREST)
            real_img = np.array(real_img)
            img = np.array(img)
            top = 0
            left = 0
#             img, top, left = np.array(img_resize(real_img,320))
            image_np = img.copy()
            input_tensor = np.array(np.expand_dims(img,0), dtype=np.float32)
            input_index = interpreter.get_input_details()[0]["index"]
            interpreter.allocate_tensors()
            interpreter.set_tensor(input_index, input_tensor)
            interpreter.invoke()
            output_details1 = interpreter.get_output_details()[0]
            output_details2 = interpreter.get_output_details()[1]
            output_details3 = interpreter.get_output_details()[3]
            detection_scores = np.squeeze(interpreter.get_tensor(output_details1['index']))
            detection_boxes = np.squeeze(interpreter.get_tensor(output_details2['index']))
            detection_classes = np.squeeze(interpreter.get_tensor(output_details3['index'])).astype(np.int64)

            res_det = np.where(detection_scores>select_min_score/100)

            if len(res_det[0]) > 0:
                label_id_offset = 1
                image = img.copy()
                imup = []
                plate_scores = []
                ind = 0
                (im_height,im_width,ch) = real_img.shape
                detection_boxes[:,1] += left/im_width
                detection_boxes[:,3] += left*2/im_width
                detection_boxes[:,0] += top/im_height
                detection_boxes[:,2] += top*2/im_height
                ymin, xmin, ymax, xmax= detection_boxes[0]
                bbox = np.array([xmin * im_width, xmax * im_width,
                            ymin * im_height, ymax * im_height]).astype(int)
                max_center = bbox[1]-bbox[0] + bbox[3]-bbox[2]
                max_score = 0
                for i in res_det[0]:
                    ymin, xmin, ymax, xmax = detection_boxes[i]
                    bbox = np.array([xmin * im_width, xmax * im_width,
                            ymin * im_height, ymax * im_height]).astype(int)
                    center = bbox[1]-bbox[0] + bbox[3]-bbox[2]
                    if detection_scores[i] > max_score:
                        max_score = detection_scores[i]
                        max_center = center
                    if np.linalg.norm(max_center - center) != 0 and np.linalg.norm(max_center - center) < 120:
                        detection_boxes = np.delete(detection_boxes, i, 0)
                        detection_classes = np.delete(detection_classes, i, 0)
                        detection_scores = np.delete(detection_scores, i, 0)
                        continue
                    plate = real_img[bbox[2]:bbox[3],bbox[0]:bbox[1]].copy()
                    best_angle,plate=correct_skew(plate, delta=1, limit=25)
                    imup.append(plate)
                    plate_scores.append(detection_scores[i])
                if len(imup) > 0:
                    viz_utils.visualize_boxes_and_labels_on_image_array(
                        real_img,
                        detection_boxes,
                        detection_classes+label_id_offset,
                        detection_scores,
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=5,
                        line_thickness=10,
                        min_score_thresh=(select_min_score/100),
                        agnostic_mode=False)
                if len(imup) > 0:
                    col2.header("Detected Image")
            else:
                col2.header("Don't detect plate")
            col2.image(real_img)
            end_t = time.time()
            st.markdown(f"Time used for detecting {round(end_t-start_t, 1)} s")
            if len(res_det[0]) > 0:
                if len(imup) > 0:
                    st.header("Detections")
                    col_pl1, col_pl2 = st.columns(2)
                    st.header("Detected Text")
                    col_tx1, col_tx2 = st.columns(2)
                for i in range(len(imup)):
                    cur_plate = imup[i]
                    platetext = reader.readtext(cur_plate)
                    platetext_pytes = pt.image_to_string(cur_plate)
                    if i % 2 == 0:
                        col_pl1.image(cur_plate, width = 300)
                        col_pl1.write("This is plate by " + str(round(plate_scores[i]*100)) +" %")
                        col_pl1.write("Pytesseract detect " + platetext_pytes)
                        for ind, text in enumerate(platetext):
                            col_tx1.write(str(ind+1) + ". " + text[1])
                    else:
                        col_pl2.image(cur_plate, width = 300)
                        col_pl2.write("This is plate by " + str(round(plate_scores[i]*100)) +" %")
                        col_pl2.write("Pytesseract detect " + platetext_pytes)
                        for ind, text in enumerate(platetext):
                            col_tx2.write(str(ind+1) + ". " + text[1])

