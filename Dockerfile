# Start your image with a node base image
FROM python:3.11.10

# The /app directory should act as the main application directory
WORKDIR /app

# Install Python dependencies
# Copy all local files into the container
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

# Run main federate starting point
CMD ["python", "ParaEMT.py"]
