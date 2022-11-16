import cv2 as cv
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import os



def our_team():
    st.markdown("<h1 style='text-align: center;'> Our Team </h1>", unsafe_allow_html=True)
    st.write("Our team consists of four people. Our team members are Sergey Xachatryan, Albert Meyroyan, Hayk Karapetyan and Georgi Sahakyan.\
             Intoduction about us below")

    Sergey = Image.open('our_pictures/Sergey.jpg').resize((400,450))
    Albert = Image.open('our_pictures/Albert.jpg').resize((400,450))
    Hayk = Image.open('our_pictures/Hayk.jpg').resize((400,450))
    Georgi =Image.open('our_pictures/Georgi.jpg').resize((400,450))
    
    col1, col2 = st.columns(2)

    col1.image(Sergey)
    col1.write('Sergey is responsable for Data Analys and SSD mobilnet model')
    col2.image(Albert)
    col2.write('Albert is responsable for Data-preprocessing and Stremlit web application')

    col3,col4 = st.columns(2)

    col3.image(Hayk)
    col3.write('Hayk is responsable for SSD mobilnet model  and Streamlit web application')
    col4.image(Georgi)
    col4.write('Georgi is responsible for data collection,data-preprocessing and data vizualization')