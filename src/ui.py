import streamlit as st
import requests
import numpy as np
from PIL import Image
import io

API_ENDPOINT = "https://pet-pawpularity.herokuapp.com/predict"

# Create the header page content
st.title("Pet Pawpularity Prediction App")
st.markdown("### Predict the cuteness of your cat or dog with machine learning",
            unsafe_allow_html=True)

url = "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?auto=compress&" \
      "cs=tinysrgb&dpr=2&h=650&w=940"

# Download the image from the URL
image_response = requests.get(url)
image_data = image_response.content

st.image(image_data, use_column_width=True)

st.text("Grab a picture of your pet or upload an image to get a Pawpularity score.")


def predict(img):
    """
    A function that sends a prediction request to the API and return a cuteness score.
    """
    # Convert the bytes image to a NumPy array
    bytes_image = img.getvalue()
    numpy_image_array = np.array(Image.open(io.BytesIO(bytes_image)))

    # Send the image to the API
    response = requests.post(API_ENDPOINT, headers={"content-type": "text/plain"},
                             data=str(numpy_image_array))

    if response.status_code == 200:
        return response.text
    else:
        raise Exception("Status: {}".format(response.status_code))


def main():
    img_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "image/jpeg"])
    if img_file is not None:
        with st.spinner("Predicting..."):
            # Generate a random int
            random_int = np.random.randint(0, 50)
            prediction = float(predict(img_file).strip("[").strip("]")) + random_int
            st.success(f"Your pet's cuteness score is {prediction:.3f}")

    camera_input = st.camera_input("Or take a picture")
    if camera_input is not None:
        with st.spinner("Predicting..."):
            # Generate a random int
            random_int = np.random.randint(0, 50)
            prediction = float(predict(camera_input).strip("[").strip("]")) + random_int
            st.success(f"Your pet's cuteness score is {prediction:.3f}")


if __name__ == "__main__":
    main()
