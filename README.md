# Air Quality Monitoring with IoT and Machine Learning using Google Cloud Platform

## Project Overview
This project demonstrates an IoT-based air quality monitoring system that integrates real-time sensor data with Google Cloud Platform (GCP) for processing, storage, and visualization. The data is analyzed using a Random Forest Classifier to predict air quality levels (Good, Moderate, Poor).

### Key Features:
   Real-Time Data Streaming: Using MQTT protocol for sensor data transmission.
   Data Storage and Analysis: Leveraging GCP BigQuery for data storage and preprocessing.
   Machine Learning: Training and deploying a Random Forest Classifier for air quality prediction.
   Visualization: Graphical representation of air quality distribution.

### Scripts
pubs.py
Publishes sensor data to an MQTT broker from a dataset stored in Google Cloud Storage (GCS) where data sensor simulation takes place.

subs_data.py
Subscribes to MQTT messages, processes the data, and inserts it into BigQuery for analysis and storage.

steps1_E.py
Handles data loading from BigQuery, machine learning model training, predictions, and air quality visualization.`


