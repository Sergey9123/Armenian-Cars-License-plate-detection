from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv
import streamlit as st
import matplotlib
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
from sklearn.manifold import TSNE
from tqdm import tqdm
from clustimage import Clustimage
import glob

@st.cache
def check_data(path):
    images = os.listdir(path)
    img_arr = []

    for i in range(9):
        image = images[i]
        if (".jpg" or ".png") in image:
            cur_img = cv.imread(path + image)
            cur_img = cv.cvtColor(cur_img,cv.COLOR_BGR2RGB)
            cur_img=cv.resize(cur_img,(200,200))
            img_arr.append(cur_img)
    return img_arr

def draw_image(img_arr):
    fig = plt.figure(figsize=(5., 5.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     )
    for ax, im in zip(grid, img_arr):
        ax.imshow(im)
        ax.axis("off")
    return fig


def draw_chart():
    count_of_cars = np.array([196, 5156, 158])
    mylabels = ["Truck", "Car", "Bus"]
    explode = [0.1, 0.1, 0.1]

    fig1, ax1 = plt.subplots()
    ax1.pie(count_of_cars, explode=explode, labels=mylabels, autopct='%1.1f%%',
            shadow=True)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    return fig1
    

def data_analys():
    
    st.markdown("<h1 style='text-align: center;'> Introduction </h1>", unsafe_allow_html=True)
    st.write("Car Data contain nearly about 5200 images. In this Notebook We are going to analyse and visualize data from Armenia Used Car data. We have choosen this data because there are no good work on this data and it is good chance to show our little skills on Data Analysis. We hope you will gain some insight from our work.")


    st.markdown("<h1 style='text-align: center;'> Data collection and Pre-processing </h1>", unsafe_allow_html=True)
    st.write("In this phase we are going to load required libraries , import data and visualize the dataset.")


    path = '../data/img_sm/'
    img_arr = check_data(path)
    with st.spinner('Loading images...'):
        fig=draw_image(img_arr[0:9])
        st.pyplot(fig)

    st.markdown("<h1 style='text-align: center;'> Distribution of detection </h1>", unsafe_allow_html=True)


    fig2=draw_chart()
    st.pyplot(fig2)

    st.subheader("Using VGG16 model we find out that 93.6% of collected data were cars, 3.6% trucks and 2.9% buses.")



















