#Note 
No Local Dataset Required: The dataset is provided by tensorflow-datasets and downloaded automatically inside the container.
Dependencies are Pre-configured: The Dockerfile ensures all necessary libraries (tensorflow, tensorflow-datasets, cloudml-hypertune) are included in the container.
Customizable Parameters: The project supports tuning multiple hyperparameters such as the number of neurons, learning rate, and momentum.
Runs on Google Cloud: The container is pushed to Google Container Registry (GCR) and deployed on Vertex AI for training.
