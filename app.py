import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Video Cloner",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Video Cloner")
st.caption("Speechify-branded video creation pipeline")

mode = st.sidebar.radio(
    "Mode",
    ["Single Job", "Clone Batch", "Persona Batch", "Producer"],
    index=0,
)

if mode == "Single Job":
    from ui.single_job import render
    render()
elif mode == "Clone Batch":
    from ui.batch_job import render
    render()
elif mode == "Persona Batch":
    from ui.persona_batch import render
    render()
else:
    from ui.producer import render
    render()
