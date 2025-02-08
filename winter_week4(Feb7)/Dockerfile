# Use a newer TensorFlow GPU base image with an updated Python version
FROM tensorflow/tensorflow:2.8.0-gpu

# Set the working directory to /app
WORKDIR /app

# Upgrade pip and install required Python libraries
RUN pip install --upgrade pip && \
    pip install cloudml-hypertune pandas numpy scikit-learn datasets

# Copy your trainer code (including task.py) into the container
COPY trainer /app/trainer

# Set the entry point to run your training module
ENTRYPOINT ["python", "-m", "trainer.task"]
