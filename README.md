# RAMP starting kit on the bike counters dataset - Bike Traffic Prediction in Paris

## Overview
This project aims to predict bike traffic in Paris using simple linear regression and display the predictions through a Streamlit web application. 
The application provides insights into bike traffic trends and predictions based on historical data, as well as the failure of simple linear regression on such datasets.

## Table of Contents
- [Introduction](#introduction)
- [Exploratory Data Analysis (EDA)](#eda)
- [Linear Regression](#linear-regression)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction <a name="introduction"></a>
Paris has a vibrant bike culture, and understanding bike traffic patterns is essential for city planning and infrastructure development. This project leverages machine learning techniques to predict bike traffic at various counters throughout the city. The predictions are visualized using Streamlit, making the results accessible and user-friendly.

## Exploratory Data Analysis (EDA) <a name="eda"></a>
The Exploratory Data Analysis section of the application allows users to explore the raw data, view a map of all counters, and visualize bike count trends for one counter.

### Raw Data
You can view the raw training data by checking the "Show raw train data" checkbox. This provides an overview of the dataset.

### Map of All Counters
A map of all counters is displayed, making it easy to visualize the distribution of bike counters in Paris.

### Bike Count Trends
The application provides plots showing the bike count trends for one counter: Totem 73 boulevard de Sébastopol S-N, helping users understand historical patterns.

## Linear Regression <a name="linear-regression"></a>
In this section, the project uses linear regression to predict bike traffic. Key steps include encoding date information, preparing the dataset, and training a linear regression model. Predictions are visualized for two counters.

### Data Encoding
Date information is encoded into separate features such as year, month, day, weekday, and hour to be used in the model.

### Linear Regressor
A Ridge regression model is used for prediction. It is trained on the encoded training data.

### Visualization
Predictions for two counters are visualized, allowing users to compare actual and predicted bike counts.
Counters were chosen to show one underestimation, and one overestimation.

## Requirements <a name="requirements"></a>

To install requirements:

```setup
pip install -r requirements.txt
```

## Usage <a name="usage"></a>
1. Launch the dockerfile to clone project and create an image
2. Launch a container for the image

