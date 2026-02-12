# Shree Kubernetes Quick Reference

## 🚀 Quick Start (Local Testing)

```bash
# 1. Build image
docker build -t shree:latest .

# 2. Test locally
docker run -e PROMETHEUS_ENABLED=true -p 8000:8000 shree:latest
curl http://localhost:8000/metrics | grep shree

# 3. Deploy to k8s (automated)
cd deploy/k8s
./deploy.sh

# 4. Check status
kubectl get pods -l app=shree
kubectl logs -f deployment/shree
```

## 📊 Prometheus Queries (PromQL)

### Live Bar Age
```promql
shree_live_bar_age_seconds{timeframe="1m"}
```

### Stale Episode Active
```promql
shree_stale_episode_active{symbol="MES"}
```

### Stale Blocks Rate (5min)
```promql
rate(shree_stale_live_bars_blocks_total[5m])
```

### Canceled Entries Rate
```promql
rate(shree_pending_entry_orders_canceled_total{reason="STALE_LIVE_BARS"}[5m])
```

### Decision Distribution
```promql
sum by (action) (rate(shree_decisions_total[5m]))
```

### Total Stale Blocks (1h)
```promql
increase(shree_stale_live_bars_blocks_total[1h])
```

### Cancellation Outcomes
```promql
sum by (outcome) (rate(shree_cancel_entries_calls_total[5m]))
```

## 🔧 Common Commands

### Deployment
```bash
# Apply manifests
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/deployment-with-secrets.yaml
kubectl apply -f deploy/prometheus/service.yaml
kubectl apply -f deploy/prometheus/servicemonitor.yaml

# Check status
kubectl get all -l app=shree
kubectl rollout status deployment/shree

# View logs
kubectl logs -f deployment/shree
kubectl logs deployment/shree --tail=100
```

### Debugging
```bash
# Pod details
kubectl describe pod -l app=shree

# Execute into pod
kubectl exec -it deployment/shree -- /bin/bash

# Port-forward metrics
kubectl port-forward svc/shree-metrics 8000:8000

# Test metrics from inside cluster
kubectl run curl --image=curlimages/curl -it --rm -- \
  curl http://shree-metrics:8000/metrics
```

### Updates
```bash
# Update image
kubectl set image deployment/shree shree=shree:v2

# Watch rollout
kubectl rollout status deployment/shree

# Rollback
kubectl rollout undo deployment/shree

# Restart pods
kubectl rollout restart deployment/shree
```

### Secrets Management
```bash
# Create secret
kubectl create secret generic shree-secrets \
  --from-literal=IBKR_HOST='127.0.0.1' \
  --from-literal=IBKR_PORT='4002'

# View secret
kubectl get secret shree-secrets -o yaml

# Update secret
kubectl delete secret shree-secrets
kubectl create secret generic shree-secrets ...

# Or edit in place
kubectl edit secret shree-secrets
```

### Scaling
```bash
# Scale up/down
kubectl scale deployment shree --replicas=2

# Autoscale (HPA)
kubectl autoscale deployment shree --min=1 --max=3 --cpu-percent=80
```

## 🎯 Prometheus Setup

### Using Prometheus Operator
```bash
# ServiceMonitor auto-discovered
kubectl get servicemonitor shree

# Check Prometheus targets
kubectl port-forward -n monitoring svc/prometheus-k8s 9090:9090
# Visit http://localhost:9090/targets
```

### Manual Prometheus Config
Add to `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'shree'
    static_configs:
      - targets: ['shree-metrics.default.svc.cluster.local:8000']
```

## 📈 Grafana Dashboard

### Import Dashboard
1. Port-forward Grafana:
   ```bash
   kubectl port-forward -n monitoring svc/grafana 3000:3000
   ```
2. Visit http://localhost:3000
3. Import `deploy/grafana/dashboard.json`

### Quick Panels
- Live Bar Age (line graph)
- Stale Episode Active (stat)
- Decision Rate by Action (stacked area)
- Stale Blocks Rate (line graph)
- Canceled Entries Rate (line graph)

## 🔔 Alert Examples

### Stale Bars Alert
```yaml
- alert: ShreeLiveBarsStale
  expr: shree_live_bar_age_seconds{timeframe="1m"} > 120
  for: 5m
  labels: { severity: warning }
  annotations:
    summary: "Live 1m bars stale"
```

### Frequent Cancellations
```yaml
- alert: ShreeFrequentEntryCancels
  expr: increase(shree_pending_entry_orders_canceled_total{reason="STALE_LIVE_BARS"}[15m]) > 3
  labels: { severity: warning }
```

## 🧹 Cleanup

```bash
# Delete all resources
kubectl delete -f deploy/k8s/deployment-with-secrets.yaml
kubectl delete -f deploy/k8s/configmap.yaml
kubectl delete -f deploy/prometheus/service.yaml
kubectl delete -f deploy/prometheus/servicemonitor.yaml
kubectl delete secret shree-secrets

# Or delete by label
kubectl delete all -l app=shree
```

## 📝 Environment Variables

### Prometheus (ConfigMap)
- `PROMETHEUS_ENABLED=true`
- `PROMETHEUS_ADDR=0.0.0.0`
- `PROMETHEUS_PORT=8000`
- `DEPLOY_ENV=prod`

### IBKR (Secret)
- `IBKR_HOST` (e.g., 127.0.0.1)
- `IBKR_PORT` (e.g., 4002)
- `IBKR_CLIENT_ID` (e.g., 1)

### Trading (ConfigMap)
- `IBKR_SYMBOL=MES`
- `MAX_MES_CONTRACTS=1`
- `RISK_PER_TRADE_USD=50`
- `DAILY_MAX_LOSS_USD=150`

## 🏷️ Useful Labels

```yaml
app: shree          # App selector
version: v1           # Version for canary
environment: prod     # Environment
```

## 🔍 Monitoring Checklist

- [ ] Metrics endpoint accessible (/metrics)
- [ ] Prometheus scraping Shree
- [ ] Grafana dashboard imported
- [ ] Alert rules configured
- [ ] Notification channels setup
- [ ] Logs aggregation configured
- [ ] Resource limits tuned
- [ ] Backup strategy defined
