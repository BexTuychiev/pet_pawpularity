import streamlit as st

API_ENDPOINT = "https://pet-pawpularity.herokuapp.com/predict"

# Create the header page content
st.title("Pet Pawpularity Prediction App")
st.markdown("### Predict the popularity of your cat or dog with machine learning",
            unsafe_allow_html=True)
with open("data/app_image.jpg", "rb") as f:
    st.image(f.read(), use_column_width=True)

st.text("Grab a picture of your pet or upload an image to get a Pawpularity score.")


def predict(img):
    """
    A function that sends a prediction request to the API and return a cuteness score.
    """
    pass

def main():
    img_file = st.file_uploader("Upload an image", type=["jpg", "png"])

    if img_file:
        st.write("Here's your pet's score!")

    camera_input = st.camera_input("Or take a picture")


if __name__ == "__main__":
    main()
