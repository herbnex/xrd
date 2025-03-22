# Use a lightweight Python 3.9 slim image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your Flask app code into the container
COPY . .

# Expose port 8080 (Cloud Run expects this port)
EXPOSE 8080

# Command to run your Flask app
CMD ["python", "server.py"]
