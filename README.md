[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/bextuychiev/pet_pawpularity/ui/src/ui.py)
<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/BexTuychiev/pet_pawpularity">
    <img src="data/app_image.jpg" alt="Logo" width="160" height="100">
  </a>

<h3 align="center">Pet Pawpularity</h3>

  <p align="center">
    Predict pet cuteness scores using machine learning.
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#usage">Detailed information</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

![](data/demo.gif)

Hi, I am Bex! I built this project to create a simple web app that would allow any user to
upload an image of their pet and get a cuteness score. The data comes from Petfinder.my
website (<a href="https://www.kaggle.com/competitions/petfinder-pawpularity-score/data">
source</a>) and contains about 10k images with labels for their cuteness. As cuteness is
such a subjective concept, the scores returned from the app are not necessarily accurate.
In fact, even the best solutions to this challenge on Kaggle are very close to a solution
that returns just random scores.

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

* [DVC](https://dvc.org/)
* [MLFlow](https://mlflow.org/)
* [Streamlit](https://streamlit.io/)
* [BentoML](https://www.bentoml.com/)
* [DagsHub](https://dagshub.com/)
* [Heroku](https://www.heroku.com/)
* [TensorFlow](https://www.tensorflow.org/)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Detailed info -->

## Detailed description of the project

I explain my approach to solve the project in three articles
on <a href="https://ibexorigin.medium.com/">my Medium blog</a>:

* [Part 1: Project Overview and DVC Setup](https://towardsdatascience.com/open-source-ml-project-with-dagshub-improve-pet-adoption-with-machine-learning-1-e9403f8f7711)
* [Part 2: Detailed tutorial to MLFlow and experiment tracking for the project](https://towardsdatascience.com/complete-guide-to-experiment-tracking-with-mlflow-and-dagshub-a0439479e0b9)
* [Part 3: In-depth Tutorial to deploying the project with the combination of DagsHub, BentoML, Streamlit](https://towardsdatascience.com/the-easiest-way-to-deploy-your-ml-dl-models-in-2022-streamlit-bentoml-dagshub-ccf29c901dac)

You can also try out the API for this project by sending a POST request
to <a href='https://pet-pawpularity.herokuapp.com/'>this address</a>. Please, read the
last part of the article for the details.
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Bex Tuychiev - [@BexTuychiev](https://www.linkedin.com/in/bextuychiev/) -
bex@ibexprogramming.com

Project
Link: [https://github.com/BexTuychiev/pet_pawpularity](https://github.com/BexTuychiev/pet_pawpularity)

<p align="right">(<a href="#top">back to top</a>)</p>
