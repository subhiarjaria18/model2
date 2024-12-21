import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load the pre-trained BLIP model
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# Streamlit app title
st.title("Image Description App")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate description button
    if st.button("Generate Description"):
        try:
            # Process the image and generate a description
            inputs = processor(images=image, return_tensors="pt")
            outputs = model.generate(**inputs)
            description = processor.decode(outputs[0], skip_special_tokens=True)

            # Display the description
            st.success("Generated Description:")
            st.write(description)
        except Exception as e:
            st.error(f"Error: {str(e)}")
