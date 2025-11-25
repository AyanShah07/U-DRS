# U-DRS Deployment Guide

Guide for deploying U-DRS in various environments.

---

## Table of Contents
- [Local Development](#local-development)
- [Production Server](#production-server)
- [Docker Deployment](#docker-deployment)
- [GPU Optimization](#gpu-optimization)
- [Edge Device Deployment](#edge-device-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Monitoring & Logging](#monitoring--logging)

---

## Local Development

### Prerequisites
```bash
# Python 3.8+
python --version

# CUDA 11.7+ (optional)
nvidia-smi

# Git
git --version
```

### Setup
```bash
# Clone
git clone https://github.com/yourusername/U-DRS.git
cd U-DRS

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

### Start Development Server
```bash
# API server with auto-reload
uvicorn api.main:app --reload --port 8000

# CLI inference
python run_inference.py --input samples/crack.jpg
```

---

## Production Server

### System Requirements
- **CPU**: 4+ cores
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional)
- **Storage**: 50GB for models + outputs
- **OS**: Ubuntu 20.04+ / CentOS 8+ / Windows Server 2019+

### Installation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.9
sudo apt install python3.9 python3.9-venv python3-pip -y

# Clone repository
git clone https://github.com/yourusername/U-DRS.git
cd U-DRS

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install GPU support (if available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Configure Systemd Service

Create `/etc/systemd/system/udrs.service`:

```ini
[Unit]
Description=U-DRS API Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/U-DRS
Environment="PATH=/opt/U-DRS/venv/bin"
ExecStart=/opt/U-DRS/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable udrs
sudo systemctl start udrs
sudo systemctl status udrs
```

### Nginx Reverse Proxy

Install Nginx:
```bash
sudo apt install nginx -y
```

Configure `/etc/nginx/sites-available/udrs`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 50M;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /opt/U-DRS/static;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/udrs /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### SSL with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

---

## Docker Deployment

### Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  udrs:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - DEVICE=cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

### Build and Run

```bash
# Build image
docker build -t udrs:latest .

# Run container (CPU)
docker run -p 8000:8000 -v $(pwd)/data:/app/data udrs:latest

# Run with GPU
docker run --gpus all -p 8000:8000 udrs:latest

# Using Docker Compose
docker-compose up -d
```

---

## GPU Optimization

### CUDA Setup

```bash
# Install NVIDIA drivers
sudo apt install nvidia-driver-525 -y

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Install cuDNN
# Download from NVIDIA website
sudo dpkg -i cudnn-local-*.deb
```

### PyTorch GPU

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### ONNX Runtime (Optional)

```bash
pip install onnxruntime-gpu

# Export model to ONNX
python scripts/export_to_onnx.py
```

### TensorRT (Advanced)

```bash
# Install TensorRT
pip install tensorrt

# Convert model
python scripts/convert_to_tensorrt.py
```

### Configuration

In `pipeline/config.py`:
```python
DEVICE = "cuda"
USE_MIXED_PRECISION = True
ENABLE_ONNX = True  # If using ONNX
```

---

## Edge Device Deployment

### Jetson Nano / Xavier

```bash
# JetPack SDK (includes CUDA, cuDNN)
sudo apt install nvidia-jetpack

# Install PyTorch for Jetson
wget https://nvidia.box.com/shared/static/[jetson-torch-wheel].whl
pip install [jetson-torch-wheel].whl

# Optimize for edge
# Use smaller models
DEPTH_MODEL_TYPE = "MiDaS_small"
INPUT_SIZE = (320, 240)
```

### Raspberry Pi

```bash
# CPU-only mode
DEVICE = "cpu"

# Use lightweight models
# Consider quantization
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Model Optimization

```python
# Quantization (INT8)
import torch.quantization as quantization

model_quantized = quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

---

## Cloud Deployment

### AWS EC2

**Instance Type**: `g4dn.xlarge` (GPU) or `c5.2xlarge` (CPU)

```bash
# Launch instance
aws ec2 run-instances \
  --image-id ami-xxxxxxxxx \
  --instance-type g4dn.xlarge \
  --key-name your-key \
  --security-groups udrs-sg

# Install NVIDIA drivers
sudo apt install nvidia-driver-470 -y

# Deploy U-DRS (follow Production Server steps)
```

### Google Cloud Platform

**Instance Type**: `n1-standard-4` with NVIDIA T4

```bash
# Create instance
gcloud compute instances create udrs-instance \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2004-lts

# Install GPU drivers
sudo /opt/deeplearning/install-driver.sh
```

### Azure

**VM Size**: `Standard_NC6` (Tesla K80)

```bash
# Create VM
az vm create \
  --resource-group udrs-rg \
  --name udrs-vm \
  --size Standard_NC6 \
  --image UbuntuLTS

# Install drivers
sudo apt install nvidia-driver-470 -y
```

### Kubernetes

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: udrs-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: udrs
  template:
    metadata:
      labels:
        app: udrs
    spec:
      containers:
      - name: udrs
        image: udrs:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: udrs-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: udrs
```

Deploy:
```bash
kubectl apply -f k8s-deployment.yaml
```

---

## Monitoring & Logging

### Prometheus + Grafana

Install Prometheus:
```bash
# Add metrics endpoint to API
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

### CloudWatch (AWS)

```python
import boto3
cloudwatch = boto3.client('cloudwatch')

# Log custom metrics
cloudwatch.put_metric_data(
    Namespace='UDRS',
    MetricData=[
        {
            'MetricName': 'InferenceTime',
            'Value': timing['total'],
            'Unit': 'Seconds'
        }
    ]
)
```

### Log Aggregation

Use ELK Stack or Loki:
```bash
# Docker Compose with Loki
docker-compose -f docker-compose-logging.yml up -d
```

---

## Performance Benchmarking

```bash
# Apache Bench
ab -n 100 -c 10 -p image.jpg -T multipart/form-data \
  http://localhost:8000/api/analyze

# Load testing
pip install locust
locust -f tests/load_test.py
```

---

## Security Checklist

- [ ] Enable HTTPS (SSL/TLS)
- [ ] Implement API key authentication
- [ ] Set up firewall rules (UFW/iptables)
- [ ] Configure rate limiting
- [ ] Enable CORS with specific origins
- [ ] Regular security updates
- [ ] Secure API keys in environment variables
- [ ] Implement input validation
- [ ] Use secrets management (AWS Secrets Manager, Vault)

---

## Backup & Recovery

```bash
# Backup data directory
tar -czf udrs-backup-$(date +%Y%m%d).tar.gz data/

# Automated backups
0 2 * * * /usr/bin/tar -czf /backups/udrs-$(date +\%Y\%m\%d).tar.gz /opt/U-DRS/data/
```

---

## Troubleshooting

### Out of Memory (OOM)
- Reduce `INPUT_SIZE` in config
- Use smaller depth model (`MiDaS_small`)
- Limit concurrent requests

### Slow Inference
- Check GPU utilization: `nvidia-smi`
- Enable mixed precision
- Use ONNX/TensorRT
- Profile with `cProfile`

### Model Loading Failures
- Check disk space
- Verify internet connection (models auto-download)
- Check PyTorch version compatibility

---

**For support, visit: [GitHub Issues](https://github.com/yourusername/U-DRS/issues)**
