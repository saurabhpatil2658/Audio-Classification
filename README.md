# Audio-Classification
UrbanSound8K Audio Classification
This project aims to classify urban sounds into predefined categories using the UrbanSound8K dataset. The dataset consists of 8732 labeled sound excerpts (4 seconds each) collected from various urban environments.

Getting Started
Follow these instructions to set up and run the project on your local machine.

Prerequisites
Make sure you have the following libraries installed:
Librosa,Scipy,Tensorflow,Pandas

You can install them using pip:
pip install librosa scipy tensorflow pandas

Data Preparation:
Dataset Download: Download the UrbanSound8K dataset from https://urbansounddataset.weebly.com/download-urbansound8k.html

Data Organization: Extract the downloaded dataset and place it in the data directory.
Metadata Load: Load metadata using the provided CSV file (UrbanSound8K.csv) using Pandas.

Data Visualization:
Visualize the dataset to gain insights into the distribution of different classes and sample audio clips.

Data Preprocessing:
  Feature Extraction: Extract features from audio samples. Here, we'll use Mel-Frequency Cepstral Coefficients (MFCCs).
  Data Splitting: Split the dataset into training, validation, and test sets.
  
Model Building
Model Architecture: Design a neural network architecture for audio classification.
Model Training: Train the model on the training data.
Model Prediction: Use the trained model to make predictions on test audio data.
Testing: Evaluate the model's performance on the test set using accuracy, precision, recall, and F1-score metrics.
Usage
data_preparation.ipynb: Jupyter notebook for data preparation.
data_visualization.ipynb: Jupyter notebook for data visualization.
feature_extraction.ipynb: Jupyter notebook for feature extraction.
model_training.ipynb: Jupyter notebook for model training and evaluation.
Acknowledgments
This project utilizes the UrbanSound8K dataset provided by Urban Sound Dataset.
