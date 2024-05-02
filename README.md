# Customer Segmentation Model with Web Interface

This project demonstrates a simple Flask application that integrates a web interface with a K-means clustering model for customer segmentation. The model is serialized using a pickle file, allowing for easy loading and prediction within the Flask app.

## Features

- **Flask Application**: A lightweight web application framework used to serve the predictive model through a user-friendly web interface.
- **Docker Integration**: Contains both Dockerfile and docker-compose.yml files for building and deploying the application in containerized environments, ensuring consistency across different setups.
- **K-means Clustering Model**: Utilizes the K-means algorithm to segment customers based on their behaviors and attributes, which is vital for targeted marketing strategies.
- **Model Serialization**: The K-means model is serialized into a pickle file, facilitating the model's deployment and usage in the web application.

## Quick Start

To get the application running locally:

1. **Clone the Repository**
   ```bash
   git clone [repository-url]
   cd deploy_ML_app

2. **Build the Docker Container**
   ```bash
   docker-compose up --build

3. **Access the Web Interface**

- Open your web browser and navigate to http://localhost:5000 to interact with the model through the web interface.