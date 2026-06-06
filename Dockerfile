FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for netCDF4, h5py, and other scientific packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    libnetcdf-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies from requirements.txt
# Also installing fastapi, uvicorn, pandas, and pydantic which are used in the app but missing from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi \
    uvicorn \
    pandas \
    pydantic

# Copy the rest of the application code
COPY . .

# Point the app at the shipped data subset (the full 40 GB data/ is .dockerignored)
ENV DATA_PATH="data_subset/*.nc"

# Expose the port the app runs on (Hugging Face Spaces expects 7860)
EXPOSE 7860

# Command to run the application. Uses $PORT if the host sets one (e.g. Render),
# otherwise defaults to 7860 to match Hugging Face Spaces / app_port.
CMD uvicorn fastapi_app:app --host 0.0.0.0 --port ${PORT:-7860}
