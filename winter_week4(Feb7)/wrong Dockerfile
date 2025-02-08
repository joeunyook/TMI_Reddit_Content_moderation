# Use TensorFlow GPU base image
FROM tensorflow/tensorflow:2.12.0-gpu

# Set the working directory
WORKDIR /

# Install required Python libraries
RUN pip install cloudml-hypertune tensorflow-datasets

# Copy the trainer code to the Docker image
COPY trainer /trainer

# Set up the entry point to invoke the trainer
ENTRYPOINT ["python", "-m", "trainer.task"]