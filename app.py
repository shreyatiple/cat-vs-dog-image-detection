import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------
# Page Config (must be first)
# ---------------------------------
st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐾",
    layout="centered"
)

IMG_SIZE = 150


# ---------------------------------
# Load Model only once
# ---------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cat_dog_classifier_model.h5")

model = load_model()


# ---------------------------------
# Custom CSS (for attractive UI)
# ---------------------------------
st.markdown("""
<style>
.big-title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
}
.sub-title {
    text-align:center;
    color:gray;
}
.result-box {
    padding:15px;
    border-radius:10px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------
# Header
# ---------------------------------
st.markdown('<p class="big-title">🐶 Dog vs Cat Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload an image and AI will predict the animal</p>', unsafe_allow_html=True)

st.divider()


# ---------------------------------
# Upload Section
# ---------------------------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])


if uploaded_file:

    col1, col2 = st.columns(2)

    # Show image
    img = Image.open(uploaded_file)

    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    # ---------------------------------
    # Prediction
    # ---------------------------------
    with col2:

        if st.button("🔍 Predict", use_container_width=True):

            with st.spinner("Analyzing image..."):

                # preprocessing
                img_resized = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)[0][0]

            dog_prob = float(prediction)
            cat_prob = 1 - dog_prob

            st.write("### Confidence")

            st.progress(int(max(dog_prob, cat_prob) * 100))

            # result display
            if dog_prob > 0.5:
                st.success(f"🐶 DOG detected ({dog_prob*100:.2f}% confident)")
            else:
                st.info(f"🐱 CAT detected ({cat_prob*100:.2f}% confident)")


st.divider()
st.caption("Built using TensorFlow + Streamlit")
