# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# Specifically, copy the src directory and the models directory
COPY ./src /app/src
COPY ./models /app/models
COPY ./sample_data.csv /app/sample_data.csv

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable (optional, if needed by your app)
# ENV NAME World

# Run main.py when the container launches
# The command needs to be able to find main.py within the src directory
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

