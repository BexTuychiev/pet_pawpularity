import requests
from skimage.io import imread

endpoint = "http://127.0.0.1:3000/predict"

# Load a sample image
img = imread("data/raw/train/0a0da090aa9f0342444a7df4dc250c66.jpg")

response = requests.post(endpoint, headers={"content-type": "text/plain"},
                         data=str(img))
