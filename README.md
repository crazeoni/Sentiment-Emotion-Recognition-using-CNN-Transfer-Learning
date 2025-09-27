# Sentiment-Emotion-Recognition-using-CNN-Transfer-Learning
Project Overview

This project implements a Convolutional Neural Network (CNN) with transfer learning to classify human emotions from facial images.
The model was trained and fine-tuned on the FER-2013 dataset, recognizing the following classes:

Anger

Disgust

Fear

Happy

Pain

Sad

The goal was to explore deep learning methods for facial emotion recognition and evaluate model performance across domains.

### Tech Stack

Python 3.10+

TensorFlow / Keras

NumPy, Pandas, Matplotlib, Seaborn

UMAP / t-SNE for embedding visualization

Google Colab (GPU runtime)

### Features

Custom CNN baseline and transfer learning with pre-trained models (EfficientNet, ResNet).

Data preprocessing, augmentation, and stratified train-test split.

Class balancing with weighted loss.

Performance evaluation with accuracy, F1-score, confusion matrix.

UMAP visualization of embeddings across layers.

Domain generalization test on external images (/new_domain/).

### Results
Model Performance

Baseline CNN Accuracy: XX%

Transfer Learning Accuracy (ResNet/EfficientNet): XX%

F1-Score (Weighted): XX

### Visualizations

Confusion Matrix:


UMAP Embeddings (Final Layer):


Sample Misclassifications:
