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
print(f"Collected {len(images)} images from Google")

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
