
# Successful setup for hyperparameter tuning

This folder contains the fully working Dockerfile and insult_gru.py file for hyperparameter tuning in a Vertex AI custom training job environment. The overall workflow and setup follow the Google Cloud Documentation here: https://cloud.google.com/vertex-ai/docs/training/custom-training-methods and you’ll find this document and the tutorial video very useful for getting a high-level understanding of how hyperparameter tuning works:https://cloud.google.com/vertex-ai/docs/training/hyperparameter-tuning-overview

However, regarding the setup for hyperparameter tuning, the Google Cloud documentation is not easy to follow, and the tutorial video overlooks several things. The major issues we encountered can be summarized as follows:

### ◉ Limited Resources:
We don't have as many resources (limited quotas for Compute Engine API, GPU, etc.) as the video suggests.

### ◉ Outdated Documentation and Versions:
The video and documentation are outdated—the versions for CUDA installation, cuDNN, and TensorFlow in our local code and Dockerfile need to be carefully investigated and updated for our setup.

### ◉External Data Dependencies:
We are using external data from Hugging Face instead of the data provided by Google Cloud, so resolving extra dependencies for dataset loading in the Dockerfile is very nuanced. The Hugging Face documentation should be read thoroughly to handle these issues.

---
For future reference, here are several aspects to be fully aware of when setting up hyperparameter tuning for your project:

In general, follow the Google Cloud documentation. However, if hyperparameter tuning results in an error (which you can view in the logs), consider the following steps:

## (1) Verify Dataset Loading in JupyterLab terminal:
Ensure that the dataset loads and the code runs correctly in your JupyterLab notebook terminal before containerizing your application. If it works locally, the code is likely fine.

## (2) Monitor Resource Quotas:
Often, the maximum quota is reached for Compute Engine or GPU availability. Always check under IAM & Admin → Quota & System Limits to ensure the project has enough resources. If not, request a quota increase. Sometimes such requests may get denied so secure resources early.

## (3) Review Dockerfile Dependencies:
If the dataset and code work locally, the issue is most likely due to Dockerfile dependencies. Make sure your Dockerfile uses:
(i) A TensorFlow GPU base image with the correct Python version.
(ii) Commands to upgrade pip and install the required libraries (for example:
pip install cloudml-hypertune pandas numpy scikit-learn datasets).
(iii) If you're using an external dataset (e.g., from Hugging Face), ensure your Docker environment’s dependencies and Python version match your local setup to load the dataset successfully.
(Compare the “wrong Dockerfile” with the Dockerfile in this folder to see the differences.)
