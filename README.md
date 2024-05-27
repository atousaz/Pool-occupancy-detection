# Project Overview: Detecting Pool Chair Occupancy

This tutorial explains a computer vision project aimed at detecting occupancy of pool chairs. The project involves collecting real and synthetic data, annotating it, and fine-tuning a YOLO model to recognize humans occupying chairs. Below are the steps and methodologies employed in the project.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Collection](#data-collection)
    - [Real Data Collection](#real-data-collection)
    - [Synthetic Data Generation](#synthetic-data-generation)
3. [Data Annotation](#data-annotation)
    - [Chair and Human Annotations](#chair-and-human-annotations)
    - [Bounding Box Annotations](#bounding-box-annotations)
4. [Model Training](#model-training)
    - [Attempting GAN Fine-Tuning](#attempting-gan-fine-tuning)
    - [Fine-Tuning YOLO](#fine-tuning-yolo)
5. [Model Training and Evaluation](#model-training-and-evaluation)
    - [Model Evaluation](#model-evaluation)
6. [Results and Analysis](#results-and-analysis)
7. [Conclusion](#conclusion)

## Introduction

In this project, the goal was to develop a vision system capable of detecting pool chair occupancy. The system can be useful for managing hotel pool areas, ensuring proper usage, and optimizing space utilization. The project involves several key steps, from data collection to model training and evaluation.

## Data Collection

### Real Data Collection

The real data was collected using Google API, which provided images of pool areas. This data served as a crucial part of the training dataset. A total of 50 images were collected using the Google API.

```python
import requests
import json

def get_google_images(api_key, search_query, num_images=50):
    url = f"https://www.googleapis.com/customsearch/v1?q={search_query}&key={api_key}&cx=YOUR_CX_ID&searchType=image&num={num_images}"
    response = requests.get(url)
    results = response.json()
    images = [item['link'] for item in results['items']]
    return images

# Example usage
api_key = 'YOUR_API_KEY'
search_query = 'hotel pool area'
images = get_google_images(api_key, search_query)
print(f"Collected {len(images)} images from Google")```

### Synthetic Data Generation
To augment the dataset, synthetic images were generated using the Stable Diffuser model from OpenAI and DALL-E. These models helped create diverse scenarios of pool chairs and human interactions, enriching the dataset. A total of 30 synthetic images were generated (10 from each model).

```python
from diffusers import StableDiffusionPipeline
import torch

def generate_synthetic_images(prompt, num_images):
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to("cuda")

    images = []
    for _ in range(num_images):
        with torch.autocast("cuda"):
            image = pipe(prompt).images[0]
        images.append(image)
    
    return images

# Example usage
prompt = "A hotel pool area with lounge chairs"
synthetic_images = generate_synthetic_images(prompt, num_images=10)
print(f"Generated {len(synthetic_images)} synthetic images using Stable Diffusion")
```
Summary of Data Collection:

Total input images: 165
Real images from Google API: 50
Synthetic images: 30
Data Annotation

Chair and Human Annotations
Annotations were created for both chairs and humans in the collected images. YOLO (You Only Look Once) was used for this task, allowing for precise bounding box annotations.

Bounding Box Annotations
Using the bounding box method, annotations were specifically created to identify humans occupying the chairs. This step involved marking the positions of humans in relation to the chairs, facilitating the training process for the occupancy detection model.

```python

import cv2
import json
import os

def annotate_image(image_path, annotations, output_path):
    image = cv2.imread(image_path)
    for annotation in annotations:
        x, y, w, h = annotation['bbox']
        label = annotation['label']
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.imwrite(output_path, image)
    return output_path

# Example usage
image_path = 'path_to_image.jpg'
annotations = [
    {'bbox': [50, 50, 100, 200], 'label': 'chair'},
    {'bbox': [200, 50, 100, 200], 'label': 'human'}
]
output_path = 'annotated_image.jpg'
annotated_image_path = annotate_image(image_path, annotations, output_path)
print(f"Annotated image saved to {annotated_image_path}")



```
Model Training

Attempting GAN Fine-Tuning
An attempt was made to fine-tune a GAN on a pretrained model to generate realistic images of pool areas. However, this approach did not yield satisfactory results, highlighting the challenges of generating realistic images without leveraging pretrained models.


```python
from keras.models import load_model

def attempt_gan_fine_tuning(model_path, data, epochs=10):
    try:
        model = load_model(model_path)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(data, epochs=epochs)
        return model
    except Exception as e:
        print(f"GAN fine-tuning failed: {e}")
        return None

# Example usage
model_path = 'pretrained_gan_model.h5'
data = load_data('path_to_training_data')
fine_tuned_model = attempt_gan_fine_tuning(model_path, data)
if fine_tuned_model:
    print("GAN fine-tuning completed")
else:
    print("GAN fine-tuning did not yield satisfactory results")



```

Fine-Tuning YOLO
The annotated images were used to create a specialized class for detecting human occupancy of chairs. YOLO was fine-tuned on this newly created class, improving its detection capabilities


```python

!pip install -U torch torchvision torchaudio
!pip install -U ultralytics

from ultralytics import YOLO

def fine_tune_yolo(data_path, epochs=10):
    model = YOLO('yolov5s.pt')  # Load pretrained YOLOv5 model
    model.train(data=data_path, epochs=epochs)  # Train the model on custom data
    return model

# Example usage
data_path = 'path_to_annotated_data.yaml'
fine_tuned_yolo = fine_tune_yolo(data_path)
print("YOLO fine-tuning completed")


```


Model Training and Evaluation

Model Evaluation
The fine-tuned YOLO model was trained and evaluated on the annotated dataset. Metrics such as accuracy, precision, recall, and F1-score were used to assess the model's performance.

```python
def evaluate_model(model, test_data_path):
    results = model.val(data=test_data_path)
    return results

# Example usage
test_data_path = 'path_to_test_data.yaml'
evaluation_results = evaluate_model(fine_tuned_yolo, test_data_path)
print(evaluation_results)

```

Results and Analysis

The results demonstrated the model's effectiveness in detecting chair occupancy. Below are the evaluation metrics extracted from the notebook


```python
Class     Images  Instances          P          R      mAP50   mAP50-95
all         33         96      0.938      0.344      0.353      0.169
all         33         96      0.969      0.344      0.355      0.346
all         33         96      0.858      0.344      0.342      0.311
all         33         96      0.846      0.344       0.34      0.318
all         33         96      0.797      0.344      0.338      0.258


```

Summary of Overall Dataset Results:

Total images processed: 165
Real images: 50
Synthetic images: 30
Total training images: 132
Total validation images: 33
Detection Results
GAN/Real:
Total images processed: 372
Total humans detected: 945
Total chairs detected: 2121
Total humans on chairs detected: 75
DALL-E:
Total images processed: 131
Total humans detected: 144
Total chairs detected: 197
Total humans on chairs detected: 41
Stable Diffuser:
Total images processed: 1064
Total humans detected: 861
Total chairs detected: 7296
Total humans on chairs detected: 89
Summary of Detection Results:

Total images processed: 1567
Total humans detected: 1950
Total chairs detected: 9614
Total humans on chairs detected: 205
