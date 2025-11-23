#!/bin/bash

# Model Serving Microservice - Google Cloud Deployment Script
# This script automates the deployment process to Google Cloud Compute Engine

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_requirements() {
    print_info "Checking requirements..."
    
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI is not installed. Please install it from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install it from: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    print_info "All requirements met!"
}

# Get configuration from user
get_config() {
    print_info "Please provide the following information:"
    
    # Get project ID
    DEFAULT_PROJECT=$(gcloud config get-value project 2>/dev/null)
    read -p "Enter your GCP Project ID [$DEFAULT_PROJECT]: " PROJECT_ID
    PROJECT_ID=${PROJECT_ID:-$DEFAULT_PROJECT}
    
    if [ -z "$PROJECT_ID" ]; then
        print_error "Project ID is required!"
        exit 1
    fi
    
    # Get zone
    read -p "Enter GCP Zone [us-central1-a]: " ZONE
    ZONE=${ZONE:-us-central1-a}
    
    # Get VM name
    read -p "Enter VM instance name [model-serving-vm]: " VM_NAME
    VM_NAME=${VM_NAME:-model-serving-vm}
    
    # Get machine type
    read -p "Enter machine type [e2-medium]: " MACHINE_TYPE
    MACHINE_TYPE=${MACHINE_TYPE:-e2-medium}
    
    # Deployment method
    echo ""
    print_info "Select deployment method:"
    echo "  1) Docker (Recommended)"
    echo "  2) Direct Python"
    read -p "Enter choice [1]: " DEPLOY_METHOD
    DEPLOY_METHOD=${DEPLOY_METHOD:-1}
    
    print_info "Configuration:"
    echo "  Project ID: $PROJECT_ID"
    echo "  Zone: $ZONE"
    echo "  VM Name: $VM_NAME"
    echo "  Machine Type: $MACHINE_TYPE"
    echo "  Deployment: $([ "$DEPLOY_METHOD" -eq 1 ] && echo 'Docker' || echo 'Direct Python')"
    echo ""
    read -p "Proceed with deployment? (y/n): " CONFIRM
    
    if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
        print_warning "Deployment cancelled."
        exit 0
    fi
}

# Set GCP project
set_project() {
    print_info "Setting GCP project to $PROJECT_ID..."
    gcloud config set project $PROJECT_ID
}

# Create VM instance
create_vm() {
    print_info "Checking if VM instance exists..."
    
    if gcloud compute instances describe $VM_NAME --zone=$ZONE &> /dev/null; then
        print_warning "VM instance '$VM_NAME' already exists."
        read -p "Do you want to use the existing VM? (y/n): " USE_EXISTING
        
        if [ "$USE_EXISTING" != "y" ] && [ "$USE_EXISTING" != "Y" ]; then
            print_error "Please delete the existing VM or choose a different name."
            exit 1
        fi
    else
        print_info "Creating VM instance '$VM_NAME'..."
        
        gcloud compute instances create $VM_NAME \
            --zone=$ZONE \
            --machine-type=$MACHINE_TYPE \
            --boot-disk-size=20GB \
            --boot-disk-type=pd-standard \
            --image-family=ubuntu-2204-lts \
            --image-project=ubuntu-os-cloud \
            --tags=http-server,https-server \
            --metadata=startup-script='#!/bin/bash
apt-get update
apt-get install -y docker.io
systemctl start docker
systemctl enable docker
usermod -aG docker $(logname)'
        
        print_info "Waiting 30 seconds for VM to initialize..."
        sleep 30
    fi
}

# Configure firewall
configure_firewall() {
    print_info "Configuring firewall rules..."
    
    if gcloud compute firewall-rules describe allow-model-serving &> /dev/null; then
        print_warning "Firewall rule 'allow-model-serving' already exists."
    else
        gcloud compute firewall-rules create allow-model-serving \
            --allow=tcp:8001 \
            --target-tags=http-server \
            --description="Allow incoming traffic on port 8001 for Model Serving API"
        
        print_info "Firewall rule created successfully!"
    fi
}

# Get VM IP
get_vm_ip() {
    print_info "Getting VM external IP address..."
    
    VM_IP=$(gcloud compute instances describe $VM_NAME \
        --zone=$ZONE \
        --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
    
    print_info "VM External IP: $VM_IP"
}

# Docker deployment
deploy_docker() {
    print_info "Starting Docker deployment..."
    
    # Build Docker image
    print_info "Building Docker image..."
    docker build -t model-serving-microservice:latest .
    
    # Tag for GCR
    print_info "Tagging image for Google Container Registry..."
    docker tag model-serving-microservice:latest \
        gcr.io/$PROJECT_ID/model-serving-microservice:latest
    
    # Configure Docker auth
    print_info "Configuring Docker authentication..."
    gcloud auth configure-docker --quiet
    
    # Push to GCR
    print_info "Pushing image to Google Container Registry..."
    docker push gcr.io/$PROJECT_ID/model-serving-microservice:latest
    
    # Deploy to VM
    print_info "Deploying to VM..."
    
    gcloud compute ssh $VM_NAME --zone=$ZONE --command="
        set -e
        echo 'Configuring Docker authentication...'
        gcloud auth configure-docker --quiet
        
        echo 'Pulling Docker image...'
        docker pull gcr.io/$PROJECT_ID/model-serving-microservice:latest
        
        echo 'Stopping and removing old container if exists...'
        docker stop model-serving 2>/dev/null || true
        docker rm model-serving 2>/dev/null || true
        
        echo 'Starting new container...'
        docker run -d \
            --name model-serving \
            --restart unless-stopped \
            -p 8001:8001 \
            gcr.io/$PROJECT_ID/model-serving-microservice:latest
        
        echo 'Checking container status...'
        docker ps | grep model-serving
    "
    
    print_info "Docker deployment completed!"
}

# Python deployment
deploy_python() {
    print_info "Starting Python deployment..."
    
    # Copy files to VM
    print_info "Copying files to VM..."
    gcloud compute scp --recurse . $VM_NAME:~/model-serving --zone=$ZONE
    
    # Setup and run on VM
    print_info "Setting up Python environment on VM..."
    
    gcloud compute ssh $VM_NAME --zone=$ZONE --command="
        set -e
        
        echo 'Installing Python and dependencies...'
        sudo apt-get update
        sudo apt-get install -y python3.11 python3.11-venv python3-pip
        
        echo 'Setting up virtual environment...'
        cd ~/model-serving
        python3.11 -m venv venv
        source venv/bin/activate
        
        echo 'Installing Python packages...'
        pip install --upgrade pip
        pip install -r requirements.txt
        
        echo 'Stopping any existing service...'
        pkill -f 'uvicorn main:app' 2>/dev/null || true
        
        echo 'Starting the service...'
        nohup uvicorn main:app --host 0.0.0.0 --port 8001 > app.log 2>&1 &
        
        sleep 3
        echo 'Checking if service is running...'
        ps aux | grep uvicorn | grep -v grep
    "
    
    print_info "Python deployment completed!"
}

# Verify deployment
verify_deployment() {
    print_info "Verifying deployment..."
    
    print_info "Waiting 10 seconds for service to start..."
    sleep 10
    
    print_info "Testing health endpoint..."
    if curl -f -s "http://$VM_IP:8001/health" > /dev/null; then
        print_info "✓ Health check passed!"
    else
        print_warning "Health check failed. Service might still be starting..."
    fi
    
    print_info "Testing root endpoint..."
    if curl -f -s "http://$VM_IP:8001/" > /dev/null; then
        print_info "✓ Root endpoint accessible!"
    else
        print_warning "Root endpoint not accessible yet."
    fi
}

# Print summary
print_summary() {
    echo ""
    echo "=========================================="
    print_info "DEPLOYMENT COMPLETE!"
    echo "=========================================="
    echo ""
    echo "Service Information:"
    echo "  • VM Name: $VM_NAME"
    echo "  • Zone: $ZONE"
    echo "  • External IP: $VM_IP"
    echo "  • API Base URL: http://$VM_IP:8001"
    echo "  • Swagger UI: http://$VM_IP:8001/docs"
    echo ""
    echo "Test Commands:"
    echo "  # Health check"
    echo "  curl http://$VM_IP:8001/health"
    echo ""
    echo "  # Get gestures"
    echo "  curl http://$VM_IP:8001/gestures"
    echo ""
    echo "SSH into VM:"
    echo "  gcloud compute ssh $VM_NAME --zone=$ZONE"
    echo ""
    
    if [ "$DEPLOY_METHOD" -eq 1 ]; then
        echo "Docker Management:"
        echo "  # View logs"
        echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --command='docker logs model-serving'"
        echo ""
        echo "  # Restart service"
        echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --command='docker restart model-serving'"
    else
        echo "Service Management:"
        echo "  # View logs"
        echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --command='tail -f ~/model-serving/app.log'"
        echo ""
        echo "  # Restart service"
        echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --command='pkill -f uvicorn && cd ~/model-serving && source venv/bin/activate && nohup uvicorn main:app --host 0.0.0.0 --port 8001 > app.log 2>&1 &'"
    fi
    
    echo ""
    echo "=========================================="
}

# Main execution
main() {
    echo ""
    echo "╔════════════════════════════════════════╗"
    echo "║  Model Serving Microservice Deployer  ║"
    echo "║     Google Cloud Compute Engine       ║"
    echo "╚════════════════════════════════════════╝"
    echo ""
    
    check_requirements
    get_config
    set_project
    create_vm
    configure_firewall
    get_vm_ip
    
    if [ "$DEPLOY_METHOD" -eq 1 ]; then
        deploy_docker
    else
        deploy_python
    fi
    
    verify_deployment
    print_summary
}

# Run main function
main

