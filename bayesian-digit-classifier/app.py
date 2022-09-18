import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from inference import calculate_overall_prediction, generate_predictions, load_model
from streamlit_drawable_canvas import st_canvas

BAYESIAN_NETWORK = load_model()

st.set_page_config(
    page_title="Bayesian Digit Classifier",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Bayesian Digit Classifier")

# Add spacing
st.markdown("#")

col1, col2 = st.columns(2)

with col1:
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


if canvas_result.image_data is not None:

    multi_preds = generate_predictions(
        numpy_image=canvas_result.image_data, bayesian_network=BAYESIAN_NETWORK, n_samples=3
    )
    print(calculate_overall_prediction(multi_preds))
    preds_df = pd.DataFrame(multi_preds.detach().numpy())
    target_classes = list(range(10))
    preds_df = pd.melt(preds_df, value_vars=target_classes).rename(
        columns={"variable": "number", "value": "proba"}
    )

    with col2:
        overall_prediction = calculate_overall_prediction(multi_preds)
        if overall_prediction:
            st.header(f"Prediction: {overall_prediction}")
        else:
            st.header(f"Not a number")

    fig = px.box(preds_df, x="number", y="proba")
    st.plotly_chart(fig, use_container_width=True)
