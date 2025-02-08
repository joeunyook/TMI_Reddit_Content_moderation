
# Successful setup for hyperparameter tuning

This folder contains the fully working Dockerfile and insult_gru.py file for hyperparameter tuning in a Vertex AI custom training job environment. The overall workflow and setup follow the Google Cloud Documentation here: https://cloud.google.com/vertex-ai/docs/training/custom-training-methods and you’ll find this document and the tutorial video very useful for getting a high-level understanding of how hyperparameter tuning works:https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview

However, regarding the setup for hyperparameter tuning, the Google Cloud documentation is not easy to follow, and the tutorial video overlooks several things. The major issues we encountered can be summarized as follows:

### ◉ Limited Resources:
We don't have as many resources (limited quotas for Compute Engine API, GPU, etc.) as the video suggests.

### ◉ Outdated Documentation and Versions:
The video and documentation are outdated—the versions for CUDA installation, cuDNN, and TensorFlow in our local code and Dockerfile need to be carefully investigated and updated for our setup.

### ◉External Data Dependencies:
We are using external data from Hugging Face instead of the data provided by Google Cloud, so resolving extra dependencies for dataset loading in the Dockerfile is very nuanced. The Hugging Face documentation should be read thoroughly to handle these issues.

