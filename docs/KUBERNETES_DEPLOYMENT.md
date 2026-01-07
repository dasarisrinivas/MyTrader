# Kubernetes Deployment Guide for MyTrader

## Prerequisites

1. **Docker** installed and running
2. **kubectl** configured with your cluster
3. **Kubernetes cluster** (local: minikube/kind, or cloud: EKS/GKE/AKS)
4. **Prometheus Operator** (optional, for ServiceMonitor)

## Step 1: Build and Push Docker Image

### Build locally
```bash
# Build the image
docker build -t mytrader:latest .

# Test locally first
docker run -e PROMETHEUS_ENABLED=true -p 8000:8000 mytrader:latest

# Verify metrics endpoint
curl http://localhost:8000/metrics
```

### Push to registry (choose one)

**Docker Hub:**
```bash
docker tag mytrader:latest YOUR_DOCKERHUB_USERNAME/mytrader:latest
docker push YOUR_DOCKERHUB_USERNAME/mytrader:latest

# Update deploy/k8s/deployment.yaml:
# image: YOUR_DOCKERHUB_USERNAME/mytrader:latest
```

**AWS ECR:**
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

# Create repository
aws ecr create-repository --repository-name mytrader --region us-east-1

# Tag and push
docker tag mytrader:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mytrader:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mytrader:latest

# Update deploy/k8s/deployment.yaml:
# image: YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mytrader:latest
```

**Google GCR:**
```bash
docker tag mytrader:latest gcr.io/YOUR_PROJECT_ID/mytrader:latest
docker push gcr.io/YOUR_PROJECT_ID/mytrader:latest

# Update deploy/k8s/deployment.yaml:
# image: gcr.io/YOUR_PROJECT_ID/mytrader:latest
```

## Step 2: Configure Secrets (IBKR Credentials)

Create a Kubernetes secret for sensitive configuration:

```bash
kubectl create secret generic mytrader-secrets \
  --from-literal=IBKR_HOST='127.0.0.1' \
  --from-literal=IBKR_PORT='4002' \
  --from-literal=IBKR_CLIENT_ID='1' \
  --from-literal=TELEGRAM_BOT_TOKEN='your-token-here' \
  --from-literal=TELEGRAM_CHAT_ID='your-chat-id'
```

Or create from file:
```bash
# Create secrets.env
cat > secrets.env <<EOF
IBKR_HOST=127.0.0.1
IBKR_PORT=4002
IBKR_CLIENT_ID=1
TELEGRAM_BOT_TOKEN=your-token-here
TELEGRAM_CHAT_ID=your-chat-id
EOF

kubectl create secret generic mytrader-secrets --from-env-file=secrets.env
rm secrets.env  # Clean up
```

Update `deploy/k8s/deployment.yaml` to use secrets (see deployment-with-secrets.yaml example).

## Step 3: Deploy to Kubernetes

### Apply all manifests
```bash
# Deploy the application
kubectl apply -f deploy/k8s/deployment.yaml

# Expose metrics service
kubectl apply -f deploy/prometheus/service.yaml

# (Optional) If using Prometheus Operator
kubectl apply -f deploy/prometheus/servicemonitor.yaml

# Check deployment status
kubectl get pods -l app=mytrader
kubectl get svc mytrader-metrics
```

### Verify deployment
```bash
# Check pod logs
kubectl logs -f deployment/mytrader

# Check pod status
kubectl describe pod -l app=mytrader

# Port-forward to test metrics locally
kubectl port-forward svc/mytrader-metrics 8000:8000

# Test metrics endpoint
curl http://localhost:8000/metrics | grep mytrader
```

## Step 4: Setup Prometheus Scraping

### Option A: Prometheus Operator (Recommended)

If you have Prometheus Operator installed:

```bash
# ServiceMonitor is already applied in Step 3
kubectl get servicemonitor mytrader

# Verify Prometheus is scraping
kubectl port-forward -n monitoring svc/prometheus-k8s 9090:9090
# Visit http://localhost:9090/targets
# Look for mytrader target
```

### Option B: Manual Prometheus Configuration

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'mytrader'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - default  # or your namespace
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: mytrader
      - source_labels: [__meta_kubernetes_pod_ip]
        action: replace
        target_label: __address__
        replacement: $1:8000
```

Or use static config:
```yaml
scrape_configs:
  - job_name: 'mytrader'
    static_configs:
      - targets: ['mytrader-metrics.default.svc.cluster.local:8000']
```

## Step 5: Setup Alerting

```bash
# Apply alert rules
kubectl apply -f deploy/prometheus/alerts.yml

# Verify alerts are loaded
kubectl port-forward -n monitoring svc/prometheus-k8s 9090:9090
# Visit http://localhost:9090/alerts
```

## Step 6: Query Metrics (Grafana/PromQL)

### Connect to Grafana
```bash
# If using Prometheus Operator stack
kubectl port-forward -n monitoring svc/grafana 3000:3000
# Visit http://localhost:3000 (admin/prom-operator)
```

### Example PromQL Queries

**Live Bar Age:**
```promql
mytrader_live_bar_age_seconds{timeframe="1m"}
```

**Stale Episode Active:**
```promql
mytrader_stale_episode_active{symbol="MES"}
```

**Stale Blocks Rate (last 5 minutes):**
```promql
rate(mytrader_stale_live_bars_blocks_total[5m])
```

**Canceled Entries Rate:**
```promql
rate(mytrader_pending_entry_orders_canceled_total{reason="STALE_LIVE_BARS"}[5m])
```

**Decision Actions Distribution:**
```promql
sum by (action) (rate(mytrader_decisions_total[5m]))
```

**Total Stale Blocks (last hour):**
```promql
increase(mytrader_stale_live_bars_blocks_total[1h])
```

## Troubleshooting

### Pod not starting
```bash
kubectl describe pod -l app=mytrader
kubectl logs -l app=mytrader --tail=100
```

### Metrics not appearing
```bash
# Check if metrics port is exposed
kubectl get svc mytrader-metrics

# Test from inside cluster
kubectl run curl --image=curlimages/curl -it --rm -- curl http://mytrader-metrics:8000/metrics

# Check Prometheus targets
kubectl port-forward -n monitoring svc/prometheus-k8s 9090:9090
# Visit http://localhost:9090/targets
```

### Image pull errors
```bash
# Check image pull policy
kubectl describe pod -l app=mytrader | grep -A5 Events

# If using private registry, create image pull secret
kubectl create secret docker-registry regcred \
  --docker-server=YOUR_REGISTRY \
  --docker-username=YOUR_USERNAME \
  --docker-password=YOUR_PASSWORD

# Add to deployment.yaml:
#   imagePullSecrets:
#     - name: regcred
```

## Production Checklist

- [ ] Build and push Docker image to registry
- [ ] Update image name in deployment.yaml
- [ ] Create secrets for IBKR credentials
- [ ] Update resource limits based on load testing
- [ ] Configure persistent volumes for data/logs (if needed)
- [ ] Setup Prometheus scraping (Operator or manual)
- [ ] Import Grafana dashboard
- [ ] Configure alert notifications (Slack/PagerDuty)
- [ ] Setup log aggregation (optional)
- [ ] Document rollback procedure

## Scaling and Updates

### Update the application
```bash
# After pushing new image
kubectl set image deployment/mytrader mytrader=mytrader:v2
kubectl rollout status deployment/mytrader

# Rollback if needed
kubectl rollout undo deployment/mytrader
```

### Scale replicas (if stateless)
```bash
kubectl scale deployment mytrader --replicas=2
```

## Clean Up

```bash
kubectl delete -f deploy/k8s/deployment.yaml
kubectl delete -f deploy/prometheus/service.yaml
kubectl delete -f deploy/prometheus/servicemonitor.yaml
kubectl delete secret mytrader-secrets
```
