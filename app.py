import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from run_model import bayesian_inference
from torchvision import transforms
import plotly.express as px
import pandas as pd
import torch

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

    multi_preds = bayesian_inference(canvas_result.image_data, process=True)

    df = pd.DataFrame(multi_preds.detach().numpy())
    df = pd.melt(df, value_vars=list(range(10))).rename(columns={
        'variable': 'number',
        'value': 'proba'
    })

    with col2:
        net_pred = torch.max(multi_preds.data, 1)[1].numpy()[0]
        st.header(f'Prediction: {net_pred}')

    fig = px.box(df, x="number", y="proba")
    st.plotly_chart(fig, use_container_width=True)



    # dm = MNISTDataModule()
    # dm.prepare_data()
    # dm.setup()

    # if st.button('Next Num:'):
    #     images, labels = next(iter(dm.train_dataloader()))

    #     TENSOR_TO_PIL = transforms.ToPILImage()
    #     idx = 0
    #     image = images[idx]
    #     st.image(TENSOR_TO_PIL(images[idx]))
    #     st.write(labels[idx], width=60)
    #     prediction = inference(image)
    #     st.write('Prediction: ', prediction)
