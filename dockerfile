# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /a
WORKDIR /AnimalGAN

# Copy the current directory contents into the container at /app
COPY . /AnimialGAN

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME AnimalGAN

# Run app.py when the container launches
CMD ["python", "main.py"]
