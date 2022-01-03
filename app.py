import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from run_model import inference

st.set_page_config(
    page_title="Bayesian Digit Classifier",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Bayesian Digit Classifier")

# Add spacing
st.markdown("#")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=24,
    stroke_color="#000000",
    background_color="#eee",
    background_image=None,
    update_streamlit=True,
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

# # Do something interesting with the image data and paths
if canvas_result.image_data is not None:

    # st.image(canvas_result.image_data)
    prediction = inference(canvas_result.image_data)
    print(prediction)
    st.write(prediction)

    st.header(f"Your number is {prediction[1]}")
