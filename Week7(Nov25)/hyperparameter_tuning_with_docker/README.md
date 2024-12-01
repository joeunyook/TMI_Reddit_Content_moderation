# Note
No Local Dataset Required: The dataset is provided by tensorflow-datasets and downloaded automatically inside the container.

Dependencies are Pre-configured: The Dockerfile ensures all necessary libraries (tensorflow, tensorflow-datasets, cloudml-hypertune are included in the container).

Customizable Parameters: The project supports tuning multiple hyperparameters such as the number of neurons, learning rate, and momentum.

Runs on Google Cloud: The container is pushed to Google Container Registry (GCR) and deployed on Vertex AI for training.

# To containerize and push the docker file to GCR follow these steps: 
Containerizing and Pushing Docker File to Google Container Registry (GCR)
Follow these steps to containerize and push your Docker file for hyperparameter tuning using Vertex AI:

Steps
1. Open JupyterLab on Vertex AI
Navigate to Vertex AI > Workbench and open JupyterLab.

2. Enable TensorFlow in JupyterLab
Install TensorFlow in your JupyterLab terminal:
pip install tensorflow

4. Set Up Docker and Build Docker Image
Run the following commands one by one in the JupyterLab terminal:

    touch Dockerfile

    mkdir trainer

    touch trainer/task.py

    PROJECT_ID="tmi-reddit-content-moderation"

    IMAGE_URI="gcr.io/$PROJECT_ID/reddit-hypertune"

    docker build ./ -t $IMAGE_URI

    docker push $IMAGE_URI
   

4. Verify Docker Push
After pushing the image, verify it by navigating to:
Vertex AI > Container Registry in the Google Cloud Console

![image](https://github.com/user-attachments/assets/d92bcd1a-be10-4615-9555-3a339b2483a0)
