# To run: streamlit run web.py
# To access: https://colab.expertsbook.org/929f18ee-a14a-4540-a945-5c7393bc579f/server/8501/
# To access(run in Term befor 'streamlit run web.py'): conda activate /projects/929f18ee-a14a-4540-a945-5c7393bc579f/conda/tf2/
import streamlit as st
from skimage import io
import streamlit.components.v1 as components
import time
from python_files.our_model import st_our_model
# from python_files.our_model_tflite import st_our_model_tflite
from python_files.other_models import st_other_models
from python_files.data_analys import data_analys
import hydralit_components as hc
from python_files.our_team import our_team

st.set_page_config(page_title="Car Plates Detection")
page_bg_img = """
<style>
.stApp {
    background-image: url("https://www.shutterstock.com/blog/wp-content/uploads/sites/5/2020/02/Usign-Gradients-Featured-Image.jpg");
    background-size: cover;
  }
</style>
"""
# https://cdn.dribbble.com/users/32512/screenshots/3032465/wave_motion_rokid_fantasy.gif
# st.markdown(page_bg_img, unsafe_allow_html=True)
st.image('stimul.jpg', width=300)


page_names_to_funcs = {
    "Our Model": st_our_model,
#     "Our Model TFlite": st_our_model_tflite,
    "Other Models": st_other_models,
    "Data Analys": data_analys,
    "Our Team": our_team
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()




