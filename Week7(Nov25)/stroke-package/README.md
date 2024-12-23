# Notes
- This is model was trained using a pre-built container, more specifically a scikit-learn container. For other types of pre-build containers, see the following link: https://cloud.google.com/vertex-ai/docs/training/pre-built-containers
- I've setup 3 storage buckets:
    - `tmi_datasets`: This contains datasets, in this case, I've uploaded a simple dataset about strokes from kaggle: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
    - `tmi_models`: When we create source distributions, in step 4, the results will be uploaded to this bucket under /vertexai/distributions after step 5
    - `tmi_outputs_us-central1`: The outputs of our training jobs will appear here in the form of model artifacts

# Setup
- If you are on mac, run `$ brew install --cask google-cloud-sdk`, we need the google cloud sdk to create a source distribition
- You may have to run `$ gcloud auth login` to authenticate your google account with the google cloud sdk
- Run `$ gcloud config set project tmi-reddit-content-moderation`
- Upload whatever dataset you want to use in the `tmi_datasets` bucket

# Procedure
1. Install local dependencies as needed (scikit-learn, numpy, etc...)
2. Note the architecture of the package, more specifically the trainer module. We have the following:
```
stroke-package
    trainer
        __init__.py
        task.py
    setup.py
```
- `__init__.py`: Signifies that the folder it belongs to is a package
- `task.py`: A package module, this is the entry point of the code. You can include training code in this module as well, or create additional modules inside your package. In our case, we would expand on this architecture to create our BERT model
- `setup.py`: The setup file specifies how to build your distribution package. Includues information such as the package name, version, as well as other packages that we may need that are not included in the GCP's pre-built training containers. Note that we have `gcsfs>=2021.4.0` because it's not included in the scikit-learn pre-built container
3. Edit the `task.py` file as to your specification
4. Create a source distribution locally with the command `python3 setup.py sdist --formats=gztar` in your main directory, which in my case, is the `stroke-package` directory. You should see `dist` and `trainer.egg-info` directories appear underneath your main directory
5. To upload you're source distribution to google cloud, run `gcloud storage cp dist/trainer-0.1.tar.gz gs://tmi_models/vertexai/distributions/` in your main directory. You should see `trainer-0.1.tar.gz` pop up in the `tmi_models` bucket under `/vertexai/distributions/`
6. Navigate to VertexAI, click training on the left sidebar, and click Train New Model. Make sure the selected region is **us-central1**
7. For training method, click no managed dataset, we will pass our dataset as an argument as denoted in `task.py`
8. If no model exists, click new model. Otherwise, click train existing model
9. More model framework, pick the appropriate pre-built container. In this case, I picked scikit-learn with a version of 0.23
10. For package location, put the directory of your source distribution, in my case, I put: `tmi_models/vertexai/distributions/trainer-0.1.tar.gz`
11. For python module, put `trainer.task` or whatever the entry point to the training code is
12. For output, I put: `tmi_outputs_us-central1/vertexai/job_outputs`
13. We need to set an argument for the directory of our dataset. We can put `--data_gcs_path=gs://tmi_datasets/healthcare-dataset-stroke-data.csv`
14. For now, we can skip hyperparameter tuning
15. For compute and pricing, I chose deploy to new worker pool and chose n1-standard-4 since the training is very light. When we have a GPU, we may need to revisit/reconfigure this step
16. Click start training and wait!
17. After the training job is complete, the model artifact should appear in `tmi_outputs_us-central1/vertexai/job_outputs`

# Resources and Links
- https://cloud.google.com/vertex-ai/docs/training/code-requirements#environment-variables
- https://cloud.google.com/vertex-ai/docs/training/create-python-pre-built-container#structure
- https://cloud.google.com/vertex-ai/docs/training/create-python-pre-built-container#create_a_source_distribution
- https://cloud.google.com/vertex-ai/docs/training/pre-built-containers