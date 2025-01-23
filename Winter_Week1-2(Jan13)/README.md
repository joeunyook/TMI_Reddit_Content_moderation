Team Update: GPU Setup and Training Progress

1. GPU Quota Secured: We've successfully increased our GPU quota on Google Cloud to 2 GPUs, which can now be attached to our Jupyter Lab notebooks.

2. GPU Setup: NVIDIA T4 GPU was successfully attached to our notebook. However, setting up CUDA drivers and cuDNN for customized Docker containers remains a time-intensive process.

3. Custom Training Status for hyperparameter tuning: While there are some issues with custom training jobs like hyperparameter tuning using customized Docker, running the code in the Cloud Shell environment works successfully with GPU.

4. Notebook Cost: The storage cost for notebooks is a lot (approximately $14 daily), notebook will only be used when needed and get deleted after being used. However, all the relevant code for processing social content data is available on our GitHub repository.

5. Dataset Update: Last week, we finalized our decision to use the "Measuring Hate Speech" dataset from Hugging Face for training. All code related to the "social-content" project now incorporates this specific dataset. link : https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech
