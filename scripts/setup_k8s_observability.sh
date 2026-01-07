#!/bin/bash
# Quick setup script for MyTrader Kubernetes observability
# This script helps you deploy Prometheus monitoring for your locally-running bot

set -e

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BOLD}MyTrader Kubernetes Observability Setup${NC}"
echo "=========================================="
echo ""

# Step 1: Check prerequisites
echo -e "${BOLD}Step 1: Checking prerequisites...${NC}"

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl not found${NC}"
    echo "Install: brew install kubectl"
    exit 1
fi
echo -e "${GREEN}✅ kubectl found${NC}"

# Check k8s cluster
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}❌ Kubernetes cluster not accessible${NC}"
    echo "Start Docker Desktop Kubernetes or minikube"
    exit 1
fi
echo -e "${GREEN}✅ Kubernetes cluster accessible${NC}"

# Check helm (optional but recommended)
if command -v helm &> /dev/null; then
    echo -e "${GREEN}✅ helm found (recommended for Prometheus install)${NC}"
    HAS_HELM=1
else
    echo -e "${YELLOW}⚠️  helm not found (install with: brew install helm)${NC}"
    HAS_HELM=0
fi

echo ""

# Step 2: Check bot metrics endpoint
echo -e "${BOLD}Step 2: Checking bot metrics endpoint...${NC}"

if curl -s --connect-timeout 2 http://localhost:8000/metrics > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Bot metrics endpoint is accessible${NC}"
else
    echo -e "${YELLOW}⚠️  Bot metrics endpoint not accessible${NC}"
    echo ""
    echo "To fix this:"
    echo "  1. Ensure bot is running: ps aux | grep run_bot.py"
    echo "  2. Set environment variables:"
    echo "     export PROMETHEUS_ENABLED=true"
    echo "     export PROMETHEUS_ADDR=0.0.0.0"
    echo "     export PROMETHEUS_PORT=8000"
    echo "     export DEPLOY_ENV=prod"
    echo "  3. Restart the bot"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

# Step 3: Get Mac mini IP
echo -e "${BOLD}Step 3: Detecting Mac mini IP address...${NC}"

MAC_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}')
echo -e "Detected IP: ${GREEN}${MAC_IP}${NC}"

# Verify this is in the scrape config
SCRAPE_CONFIG="deploy/k8s-observability/prometheus-scrape-config.yaml"
if grep -q "ip: ${MAC_IP}" "${SCRAPE_CONFIG}"; then
    echo -e "${GREEN}✅ Scrape config already has correct IP${NC}"
else
    echo -e "${YELLOW}⚠️  Scrape config has different IP${NC}"
    echo "Updating scrape config with IP: ${MAC_IP}"
    # Create backup
    cp "${SCRAPE_CONFIG}" "${SCRAPE_CONFIG}.bak"
    # Update IP (assumes line format "- ip: X.X.X.X")
    sed -i.tmp "s/- ip: [0-9.]*$/- ip: ${MAC_IP}  # Auto-updated/" "${SCRAPE_CONFIG}"
    rm "${SCRAPE_CONFIG}.tmp"
    echo -e "${GREEN}✅ Updated scrape config${NC}"
fi

echo ""

# Step 4: Install Prometheus (if not present)
echo -e "${BOLD}Step 4: Checking for Prometheus...${NC}"

if kubectl get namespace monitoring &> /dev/null; then
    echo -e "${GREEN}✅ Monitoring namespace exists${NC}"
else
    echo -e "${YELLOW}⚠️  Monitoring namespace not found${NC}"
    
    if [ $HAS_HELM -eq 1 ]; then
        echo ""
        echo "Would you like to install Prometheus using Helm?"
        echo "This will install: Prometheus + Grafana + Alertmanager"
        read -p "Install? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Installing kube-prometheus-stack..."
            helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
            helm repo update
            helm install prometheus prometheus-community/kube-prometheus-stack \
                --namespace monitoring \
                --create-namespace \
                --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
                --wait
            echo -e "${GREEN}✅ Prometheus installed${NC}"
        else
            echo -e "${YELLOW}⚠️  Skipping Prometheus install. You'll need to install it manually.${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Install Prometheus manually or install helm first${NC}"
    fi
fi

echo ""

# Step 5: Apply manifests
echo -e "${BOLD}Step 5: Applying Kubernetes manifests...${NC}"

echo "Applying scrape configuration..."
kubectl apply -f deploy/k8s-observability/prometheus-scrape-config.yaml

echo "Applying alert rules..."
kubectl apply -f deploy/k8s-observability/prometheus-rules.yaml

echo -e "${GREEN}✅ Manifests applied${NC}"

echo ""

# Step 6: Verify deployment
echo -e "${BOLD}Step 6: Verifying deployment...${NC}"

echo "Checking resources..."
kubectl get endpoints mytrader-external
kubectl get svc mytrader-external
kubectl get servicemonitor mytrader-external 2>/dev/null || echo "(ServiceMonitor requires Prometheus Operator)"
kubectl get prometheusrule mytrader-alerts 2>/dev/null || echo "(PrometheusRule requires Prometheus Operator)"

echo ""

# Step 7: Test connectivity from k8s
echo -e "${BOLD}Step 7: Testing connectivity from Kubernetes...${NC}"

echo "Running test pod to verify k8s can reach Mac mini..."
if kubectl run test-curl --rm -i --image=curlimages/curl --restart=Never -- \
    curl -s --connect-timeout 5 http://${MAC_IP}:8000/metrics | head -5 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Kubernetes can reach Mac mini metrics endpoint${NC}"
else
    echo -e "${RED}❌ Kubernetes cannot reach Mac mini metrics endpoint${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check firewall allows port 8000"
    echo "  2. Verify bot is running and exposing metrics"
    echo "  3. Check IP address is correct: ${MAC_IP}"
    echo "  4. Try manually: kubectl run test-curl --rm -it --image=curlimages/curl --restart=Never -- curl -v http://${MAC_IP}:8000/metrics"
fi

echo ""

# Step 8: Instructions for next steps
echo -e "${BOLD}✅ Setup Complete!${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Access Prometheus UI:"
echo "   kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090"
echo "   Then open: http://localhost:9090"
echo ""
echo "2. Check Targets (should show 'mytrader-external' as UP):"
echo "   http://localhost:9090/targets"
echo ""
echo "3. Check Alerts (should show MyTrader alert rules):"
echo "   http://localhost:9090/alerts"
echo ""
echo "4. Try some queries:"
echo "   up{job=~\".*mytrader.*\"}"
echo "   mytrader_live_bar_age_seconds"
echo "   rate(mytrader_decisions_total[5m])"
echo ""
echo "5. Access Grafana (if installed):"
echo "   kubectl get secret -n monitoring prometheus-grafana -o jsonpath=\"{.data.admin-password}\" | base64 --decode"
echo "   kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80"
echo "   Then open: http://localhost:3000 (user: admin)"
echo ""
echo "6. Validation script:"
echo "   ./scripts/check_metrics_reachable.sh localhost 8000"
echo ""
echo -e "📚 Full documentation: ${GREEN}docs/OBSERVABILITY_K8S.md${NC}"
echo -e "🚀 Quick start guide: ${GREEN}deploy/k8s-observability/README.md${NC}"
echo ""
echo "Your Mac mini IP: ${MAC_IP}"
echo "Metrics endpoint: http://${MAC_IP}:8000/metrics"
echo ""
