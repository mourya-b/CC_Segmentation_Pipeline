FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    timm==0.9.16 \
    monai==1.3.0 \
    albumentations==1.4.3 \
    nibabel==5.2.1 \
    pydicom==2.4.4 \
    SimpleITK==2.3.1 \
    zarr==3.0.0a5 \
    pandas==2.2.1 \
    openpyxl==3.1.2 \
    PyYAML==6.0.1

# Set working directory
WORKDIR /opt/app

# Copy source code
COPY src/ src/
COPY configs/ configs/

ENV PYTHONPATH=/opt/app