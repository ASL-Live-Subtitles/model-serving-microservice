# Model Serving Microservice - Deployment Guide

## Current Deployment

**Service URL:** http://34.132.238.32:8001  
**Swagger UI:** http://34.132.238.32:8001/docs  
**Project:** model-serving-ml  
**VM:** model-serving-vm (e2-medium, us-central1-a)  
**Cost:** ~$27/month (free with $300 trial credits)

---

## Quick Deploy to Google Cloud

### Prerequisites

- Google Cloud SDK installed
- Docker installed
- GCP project with billing enabled

### Automated Deployment

```bash
./deploy-to-gcp.sh
```

### Manual Deployment

```bash
# 1. Authenticate and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Enable APIs
gcloud services enable compute.googleapis.com containerregistry.googleapis.com

# 3. Create VM
gcloud compute instances create model-serving-vm \
    --zone=us-central1-a \
    --machine-type=e2-medium \
    --boot-disk-size=20GB \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --tags=http-server

# 4. Configure firewall
gcloud compute firewall-rules create allow-model-serving \
    --allow=tcp:8001 \
    --target-tags=http-server

# 5. Build for AMD64
docker buildx build --platform linux/amd64 -t model-serving:amd64 . --load

# 6. Save and transfer
docker save model-serving:amd64 | gzip > image.tar.gz
gcloud compute scp image.tar.gz model-serving-vm:~ --zone=us-central1-a

# 7. Deploy on VM
gcloud compute ssh model-serving-vm --zone=us-central1-a --command="
  sudo docker load < image.tar.gz
  sudo docker run -d --name model-serving --restart unless-stopped -p 8001:8001 model-serving:amd64
"

# 8. Get IP and test
VM_IP=$(gcloud compute instances describe model-serving-vm \
    --zone=us-central1-a \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
curl http://$VM_IP:8001/health
```

---

## Cloud Run Alternative

Cheaper option (~$0-2/month) for variable traffic:

```bash
gcloud run deploy model-serving \
    --source . \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8001 \
    --memory 512Mi
```

---

## Management Commands

### View logs

```bash
gcloud compute ssh model-serving-vm --zone=us-central1-a --command="sudo docker logs model-serving"
```

### Restart service

```bash
gcloud compute ssh model-serving-vm --zone=us-central1-a --command="sudo docker restart model-serving"
```

### Stop VM (save costs)

```bash
gcloud compute instances stop model-serving-vm --zone=us-central1-a
```

### Update deployment

```bash
# Rebuild image
docker buildx build --platform linux/amd64 -t model-serving:amd64 . --load
docker save model-serving:amd64 | gzip > image.tar.gz

# Transfer and redeploy
gcloud compute scp image.tar.gz model-serving-vm:~ --zone=us-central1-a
gcloud compute ssh model-serving-vm --zone=us-central1-a --command="
  sudo docker load < image.tar.gz
  sudo docker stop model-serving
  sudo docker rm model-serving
  sudo docker run -d --name model-serving --restart unless-stopped -p 8001:8001 model-serving:amd64
"
```

---

## API Endpoints

### Test endpoints

```bash
# Health check
curl http://34.132.238.32:8001/health

# Get gestures
curl http://34.132.238.32:8001/gestures

# Create gesture
curl -X POST http://34.132.238.32:8001/gestures \
  -H "Content-Type: application/json" \
  -d '{"landmarks": [[0.5, 0.6], ...], "user_id": "test"}'

# Get models
curl http://34.132.238.32:8001/models

# Register model
curl -X POST http://34.132.238.32:8001/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Classifier",
    "version": "v1.0.0",
    "description": "ASL classifier",
    "model_path": "/models/model.tflite",
    "input_shape": [42],
    "output_classes": 37,
    "model_type": "classification"
  }'
```

### Integration example

```python
import httpx

BASE_URL = "http://34.132.238.32:8001"

async def predict_gesture(landmarks, user_id=None):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/gestures",
            json={"landmarks": landmarks, "user_id": user_id}
        )
        return response.json()
```

---

## Troubleshooting

### Service not responding

```bash
# Check container status
gcloud compute ssh model-serving-vm --zone=us-central1-a --command="sudo docker ps"

# Check logs
gcloud compute ssh model-serving-vm --zone=us-central1-a --command="sudo docker logs model-serving"

# Restart
gcloud compute ssh model-serving-vm --zone=us-central1-a --command="sudo docker restart model-serving"
```

### Port issues

```bash
# Check firewall
gcloud compute firewall-rules list --filter="name=allow-model-serving"

# Verify port binding
gcloud compute ssh model-serving-vm --zone=us-central1-a --command="sudo netstat -tlnp | grep 8001"
```

### Image architecture mismatch

Always build for AMD64 when deploying to GCP VMs:

```bash
docker buildx build --platform linux/amd64 -t model-serving:amd64 . --load
```

---

## Cost Breakdown

### Compute Engine VM

- e2-medium: ~$27/month
- External IP: ~$3/month
- Disk (20GB): ~$1.70/month
- Total: ~$32/month

### Cloud Run (alternative)

- Low traffic: ~$0-2/month
- Scales to zero when idle
- Pay per request

### Free trial

New accounts get $300 credits for 90 days, covering 9-10 months of VM costs.

---

## Cleanup

```bash
# Delete VM
gcloud compute instances delete model-serving-vm --zone=us-central1-a

# Delete firewall rule
gcloud compute firewall-rules delete allow-model-serving

# Delete project (removes everything)
gcloud projects delete model-serving-ml
```
