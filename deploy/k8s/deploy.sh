#!/bin/bash
# Quick deployment script for MyTrader on Kubernetes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}MyTrader Kubernetes Deployment${NC}"
echo "================================"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: docker not found${NC}"
    exit 1
fi

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites satisfied${NC}"
echo ""

# Configuration
IMAGE_NAME=${IMAGE_NAME:-"mytrader:latest"}
NAMESPACE=${NAMESPACE:-"default"}
USE_SECRETS=${USE_SECRETS:-"false"}

echo "Configuration:"
echo "  Image: $IMAGE_NAME"
echo "  Namespace: $NAMESPACE"
echo "  Use Secrets: $USE_SECRETS"
echo ""

# Build Docker image
read -p "Build Docker image? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME .
    echo -e "${GREEN}✓ Image built${NC}"
    echo ""
fi

# Push to registry (optional)
read -p "Push image to registry? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter registry URL (e.g., docker.io/username or ECR URL): " REGISTRY_URL
    FULL_IMAGE="$REGISTRY_URL/$IMAGE_NAME"
    echo "Tagging and pushing to $FULL_IMAGE..."
    docker tag $IMAGE_NAME $FULL_IMAGE
    docker push $FULL_IMAGE
    echo -e "${GREEN}✓ Image pushed${NC}"
    
    # Update deployment file
    if [[ "$USE_SECRETS" == "true" ]]; then
        sed -i.bak "s|image:.*|image: $FULL_IMAGE|g" deploy/k8s/deployment-with-secrets.yaml
    else
        sed -i.bak "s|image:.*|image: $FULL_IMAGE|g" deploy/k8s/deployment.yaml
    fi
    echo ""
fi

# Create namespace if needed
if [[ "$NAMESPACE" != "default" ]]; then
    echo "Creating namespace $NAMESPACE..."
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    echo ""
fi

# Create secrets
if [[ "$USE_SECRETS" == "true" ]]; then
    read -p "Create/update secrets? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating secrets..."
        read -p "IBKR Host (default: 127.0.0.1): " IBKR_HOST
        IBKR_HOST=${IBKR_HOST:-"127.0.0.1"}
        
        read -p "IBKR Port (default: 4002): " IBKR_PORT
        IBKR_PORT=${IBKR_PORT:-"4002"}
        
        read -p "IBKR Client ID (default: 1): " IBKR_CLIENT_ID
        IBKR_CLIENT_ID=${IBKR_CLIENT_ID:-"1"}
        
        read -p "Telegram Bot Token (optional): " TELEGRAM_BOT_TOKEN
        read -p "Telegram Chat ID (optional): " TELEGRAM_CHAT_ID
        
        kubectl create secret generic mytrader-secrets \
            --from-literal=IBKR_HOST="$IBKR_HOST" \
            --from-literal=IBKR_PORT="$IBKR_PORT" \
            --from-literal=IBKR_CLIENT_ID="$IBKR_CLIENT_ID" \
            --from-literal=TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN" \
            --from-literal=TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID" \
            --namespace=$NAMESPACE \
            --dry-run=client -o yaml | kubectl apply -f -
        
        echo -e "${GREEN}✓ Secrets created${NC}"
        echo ""
    fi
    
    # Apply ConfigMap
    echo "Applying ConfigMap..."
    kubectl apply -f deploy/k8s/configmap.yaml -n $NAMESPACE
    echo -e "${GREEN}✓ ConfigMap applied${NC}"
    echo ""
fi

# Deploy application
echo "Deploying MyTrader..."
if [[ "$USE_SECRETS" == "true" ]]; then
    kubectl apply -f deploy/k8s/deployment-with-secrets.yaml -n $NAMESPACE
else
    kubectl apply -f deploy/k8s/deployment.yaml -n $NAMESPACE
fi
echo -e "${GREEN}✓ Deployment created${NC}"
echo ""

# Deploy Service
echo "Creating Service..."
kubectl apply -f deploy/prometheus/service.yaml -n $NAMESPACE
echo -e "${GREEN}✓ Service created${NC}"
echo ""

# Deploy ServiceMonitor (if Prometheus Operator exists)
read -p "Deploy ServiceMonitor for Prometheus Operator? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    kubectl apply -f deploy/prometheus/servicemonitor.yaml -n $NAMESPACE
    echo -e "${GREEN}✓ ServiceMonitor created${NC}"
    echo ""
fi

# Wait for deployment
echo "Waiting for deployment to be ready..."
kubectl rollout status deployment/mytrader -n $NAMESPACE --timeout=300s
echo -e "${GREEN}✓ Deployment ready${NC}"
echo ""

# Show status
echo "Deployment Status:"
echo "=================="
kubectl get pods -l app=mytrader -n $NAMESPACE
echo ""
kubectl get svc mytrader-metrics -n $NAMESPACE
echo ""

# Show logs
echo -e "${YELLOW}Recent logs:${NC}"
kubectl logs -l app=mytrader -n $NAMESPACE --tail=20
echo ""

# Port-forward instructions
echo -e "${GREEN}Deployment complete!${NC}"
echo ""
echo "To access metrics locally:"
echo "  kubectl port-forward svc/mytrader-metrics 8000:8000 -n $NAMESPACE"
echo "  curl http://localhost:8000/metrics"
echo ""
echo "To view logs:"
echo "  kubectl logs -f deployment/mytrader -n $NAMESPACE"
echo ""
echo "To check pod status:"
echo "  kubectl describe pod -l app=mytrader -n $NAMESPACE"
