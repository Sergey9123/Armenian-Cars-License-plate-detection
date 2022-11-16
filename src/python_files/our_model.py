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

configs = config_util.get_configs_from_pipeline_file('../data/models/pipeline.config')
category_index = label_map_util.create_category_index_from_labelmap('../data/models/label_map.pbtxt')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore('../data/models/ckpt-11').expect_partial()

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

def read_files():
    data = os.listdir('../data/raw/')
    data.sort()
    snaps = os.listdir('../data/snaps/')
    snaps.sort()
    return data,snaps


def st_our_model_video():
    select_min_score = st.slider('Min Score of Detecting %',min_value = 10, max_value = 100, value=20, step = 1)
    if st.button("Run Detector"):
        with st.spinner('Detecting ...'):
            start_t = time.time()
            cap = cv.VideoCapture('test_video.mp4')
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            out = cv.VideoWriter('detected_plate.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
            if (cap.isOpened()== False):
                print("Error opening video stream or file")
            while(cap.isOpened()):
                ret, img = cap.read()
                if ret == True:
                    image_np = np.array(img)

                    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
                    detections = detect_fn(input_tensor)

                    num_detections = int(detections.pop('num_detections'))
                    detections = {key: value[0, :num_detections].numpy()
                                  for key, value in detections.items()}
                    detections['num_detections'] = num_detections
                    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                    res_det = np.where(detections['detection_scores']>select_min_score/100)

                    label_id_offset = 1
                    image_np_with_detections = image_np.copy()

                    image = image_np_with_detections
                    imup = []
                    plate_scores = []
                    ind = 0
                    ymin, xmin, ymax, xmax=detections['detection_boxes'][0]
                    (im_height,im_width,ch) = image_np.shape
                    bbox = np.array([xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height]).astype(int)
                    max_center = bbox[1]-bbox[0] + bbox[3]-bbox[2]
                    max_score = 0
                    for i in res_det[0]:
                        ymin, xmin, ymax, xmax=detections['detection_boxes'][i]

                        (im_height,im_width,ch) = image_np.shape
                        bbox = np.array([xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height]).astype(int)
                        center = bbox[1]-bbox[0] + bbox[3]-bbox[2]
                        if detections['detection_scores'][i] > max_score:
                            max_score = detections['detection_scores'][i]
                            max_center = center
                        if np.linalg.norm(max_center - center) != 0 and np.linalg.norm(max_center - center) < 110:
                            detections['detection_boxes'] = np.delete(detections['detection_boxes'], i, 0)
                            detections['detection_classes'] = np.delete(detections['detection_classes'], i, 0)
                            detections['detection_scores'] = np.delete(detections['detection_scores'], i, 0)
                            continue
                        plate = image[bbox[2]:bbox[3],bbox[0]:bbox[1]].copy()
                        best_angle,plate=correct_skew(plate, delta=1, limit=25)
                        imup.append(plate)
                        plate_scores.append(detections['detection_scores'][i])
                    if len(imup) > 0:
                        viz_utils.visualize_boxes_and_labels_on_image_array(
                            image,
                            detections['detection_boxes'],
                            detections['detection_classes']+label_id_offset,
                            detections['detection_scores'],
                            category_index,
                            use_normalized_coordinates=True,
                            max_boxes_to_draw=5,
                            line_thickness=10,
                            min_score_thresh=(select_min_score/100),
                            agnostic_mode=False)
                    out.write(image)
                    if cv.waitKey(25) & 0xFF == ord('q'):
                        break
                else:
                    break
            cap.release()
            cv.destroyAllWindows()
            end_t = time.time()
            st.markdown(f"Time used for detecting {round(end_t-start_t, 1)} s")

global filenames
filenames = []

def st_our_model():
    global filenames
    data,snaps=read_files()
    if st.checkbox("Uploaded Images"):
        select_image = st.selectbox("Choose a picture",options = snaps,index = 0, key = 1)
        path = '../data/snaps/'
    else:
        select_image = st.selectbox("Choose a picture",options = data,index = 1, key = 1)
        path = '../data/raw/'
    select_min_score = st.slider('Min Score of Detecting %',min_value = 10, max_value = 100, value=20, step = 1)
    st.header("Example of detected image")
    uploaded_file = st.file_uploader("Choose a file for detecting image",["jpg","png"])
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
        img = cv.cvtColor(np.array(img),cv.COLOR_BGR2RGB)
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
        with st.spinner('Detecting ...'):
            start_t = time.time()
            IMAGE_PATH=det_img
            img = cv.imread(IMAGE_PATH)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            image_np = np.array(img)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            res_det = np.where(detections['detection_scores']>select_min_score/100)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            image = image_np_with_detections
            imup = []
            plate_scores = []
            ind = 0
            ymin, xmin, ymax, xmax=detections['detection_boxes'][0]
            (im_height,im_width,ch) = image_np.shape
            bbox = np.array([xmin * im_width, xmax * im_width,
                        ymin * im_height, ymax * im_height]).astype(int)
            max_center = bbox[1]-bbox[0] + bbox[3]-bbox[2]
            max_score = 0
            for i in res_det[0]:
                ymin, xmin, ymax, xmax=detections['detection_boxes'][i]

                (im_height,im_width,ch) = image_np.shape
                bbox = np.array([xmin * im_width, xmax * im_width,
                        ymin * im_height, ymax * im_height]).astype(int)
                center = bbox[1]-bbox[0] + bbox[3]-bbox[2]
                if detections['detection_scores'][i] > max_score:
                    max_score = detections['detection_scores'][i]
                    max_center = center
                if np.linalg.norm(max_center - center) != 0 and np.linalg.norm(max_center - center) < 110:
                    detections['detection_boxes'] = np.delete(detections['detection_boxes'], i, 0)
                    detections['detection_classes'] = np.delete(detections['detection_classes'], i, 0)
                    detections['detection_scores'] = np.delete(detections['detection_scores'], i, 0)
                    continue
                plate = image[bbox[2]:bbox[3],bbox[0]:bbox[1]].copy()
                best_angle,plate=correct_skew(plate, delta=1, limit=25)
                imup.append(plate)
                plate_scores.append(detections['detection_scores'][i])
            if len(imup) > 0:
                viz_utils.visualize_boxes_and_labels_on_image_array(
                    image,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    line_thickness=10,
                    min_score_thresh=(select_min_score/100),
                    agnostic_mode=False)

            if len(imup) > 0:
                col2.header("Detected Image")
                col2.image(image/255, use_column_width = True)
            else:
                st.header("Don't detect plate")
            end_t = time.time()
            st.markdown(f"Time used for detecting {round(end_t-start_t, 1)} s")
            if len(imup) > 0:
                st.header("Detections")
                col_pl1, col_pl2 = st.columns(2)
                st.header("Detected Text")
                col_tx1, col_tx2 = st.columns(2)
            for i in range(len(imup)):
                cur_plate = imup[i]
                platetext = reader.readtext(cur_plate)
                if i % 2 == 0:
                    col_pl1.image(cur_plate, use_column_width = True)
                    col_pl1.write("This is plate by " + str(round(plate_scores[i]*100)) +" %")
                    for ind, text in enumerate(platetext):
                        col_tx1.write(str(ind+1) + ". " + text[1])
                else:
                    col_pl2.image(cur_plate, use_column_width = True)
                    col_pl2.write("This is plate by " + str(round(plate_scores[i]*100)) +" %")
                    for ind, text in enumerate(platetext):
                        col_tx2.write(str(ind+1) + ". " + text[1])

