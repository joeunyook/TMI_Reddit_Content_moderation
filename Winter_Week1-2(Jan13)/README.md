Team Updates for the First 2 Weeks of the Winter Semester:

Successful GPU Setup and Training Progress / Finalizing Dataset

1. GPU Quota Secured: We've successfully increased our GPU quota on Google Cloud to 2 GPUs, which can now be attached to our Jupyter Lab notebooks.

2. GPU Setup: NVIDIA T4 GPU was successfully attached to our notebook. However, setting up CUDA drivers and cuDNN for customized Docker containers remains a time-intensive process.

3. Custom Training Status for hyperparameter tuning: While there are some issues with custom training jobs like hyperparameter tuning using customized Docker, running the code in the Cloud Shell environment works successfully with GPU.

4. Notebook Cost: The storage cost for notebooks is quite high (approximately $14 daily). Notebooks will only be used when needed and will be deleted after use. However, all the relevant code for processing social content data is available in our GitHub repository. If you need to test something, feel free to create a notebook, VM, or training job, but please ensure it is deleted after you finish your work.
   
6. Dataset Update: Last week, we finalized our decision to use the "Measuring Hate Speech" dataset from Hugging Face for training. All code related to the "social-content" project now incorporates this specific dataset. link : https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech
![Validation Accuracy - gpu attached](https://github.com/user-attachments/assets/d94598bc-ce43-43ae-b603-db9a7c2c1aa0)
