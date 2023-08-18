# Image Captioning Web Application using Flask, LSTM, and CNN

## Description of the Project

The Image Captioning Web Application aims to provide an interactive platform for generating captions for images automatically. The captions are generated using a deep learning model that combines Convolutional Neural Networks (CNNs) for image processing and Long Short-Term Memory (LSTM) networks for sequence processing. The application is built using the Flask web framework to allow users to upload images and receive generated captions in real-time.

### Dataset

The dataset used in this project is Flicker8k, which contains around 8000 images with corresponding captions.

![image_captioning_dataset](https://user-images.githubusercontent.com/74714252/228909136-9c5ee7a7-1281-41ee-af50-210913231f76.png)

### Dependencies

- Python 3.6 
- Flask
- TensorFlow 2.0 
- Keras 2.0 
- NumPy
- Pillow

## Data Preprocessing

The data preprocessing involves the following steps:

- Load the image dataset
- Preprocess the images using CNNs
- Preprocess the captions using LSTM

### Load the image dataset

The first step is to load the image dataset from Flicker8k. The images are resized to 224x224 pixels, and the pixel values are normalized to be in the range of [0, 1].

**Preprocess the images using CNNs**

The images are processed using a pre-trained CNN model (VGG-16) to extract features. The model is fine-tuned on the image captioning task by removing the last layer and adding a new layer to output the feature vector.

**Preprocess the captions using LSTM**

The captions are preprocessed using LSTM networks. The words are tokenized, and a vocabulary is created. The captions are padded to have the same length, and the word indices are converted to one-hot vectors.

## Model Architecture

The model architecture consists of a CNN for image processing and an LSTM for sequence processing. The image features are fed into the LSTM along with the word embeddings of the caption. The output of the LSTM is a sequence of words that form the caption.

![model](https://user-images.githubusercontent.com/74714252/228908085-319930a6-9251-4fb5-a7c5-181a1404869d.png)

## Flask Web Application

The Flask web application allows users to upload images and receive generated captions in real-time. Users can interact with the application through a user-friendly web interface.

### Usage

1. Install the required dependencies using `pip install -r requirements.txt`.
2. Run the Flask application using `python app.py`.
3. Access the web application through a web browser.
4. Upload an image, and the application will generate a caption using the trained model and display it on the screen.

### Website Link

Visit the deployed Image Captioning Web Application: [https://image-caption-generator-mx3k.onrender.com](https://image-caption-generator-mx3k.onrender.com)

## Evaluation Metric

The evaluation metric used in this project is BLEU. BLEU (Bilingual Evaluation Understudy) is a metric for evaluating the quality of generated text. It compares the generated text to a set of reference texts and computes a score based on the n-grams overlap.

## Conclusion

In this project, we have implemented an image captioning web application using Flask, LSTM, and CNN. The model has been trained on the Flicker8k dataset and evaluated using the BLEU metric. The web application provides an intuitive way for users to generate captions for new images, making it a useful tool for automatic image annotation.
