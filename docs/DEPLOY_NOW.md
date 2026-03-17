# Shree Kubernetes Deployment - Ready to Deploy

## ✅ Files Created

All deployment files are ready in your repository:

```
Shree/
├── Dockerfile                                    # Container image definition
├── deploy/
│   ├── k8s/
│   │   ├── deployment.yaml                       # Basic deployment
│   │   ├── deployment-with-secrets.yaml          # Production deployment with secrets
│   │   ├── configmap.yaml                        # Non-sensitive configuration
│   │   ├── deploy.sh                             # Automated deployment script
│   │   ├── QUICKREF.md                           # Quick reference guide
│   ├── prometheus/
│   │   ├── service.yaml                          # Metrics service
│   │   ├── servicemonitor.yaml                   # Prometheus Operator config
│   │   └── alerts.yml                            # Alert rules
│   └── grafana/
│       └── dashboard.json                        # Grafana dashboard
└── docs/
    ├── KUBERNETES_DEPLOYMENT.md                  # Complete deployment guide
    └── OBSERVABILITY.md                          # Prometheus setup guide
```

## 🚀 Deploy Now (3 Options)

### Option 1: Automated Script (Recommended for Testing)

```bash
cd /Users/svss/Documents/code/Shree

# Run interactive deployment
./deploy/k8s/deploy.sh
```

This will:
- Build Docker image
- Optionally push to registry
- Create secrets interactively
- Deploy all manifests
- Show status and logs

### Option 2: Manual Deployment (Step-by-Step)

```bash
# 1. Build Docker image
docker build -t shree:latest .

# 2. Test locally first
docker run -e PROMETHEUS_ENABLED=true -p 8000:8000 shree:latest &
curl http://localhost:8000/metrics | grep shree
docker stop $(docker ps -q --filter ancestor=shree:latest)

# 3. Create ConfigMap
kubectl apply -f deploy/k8s/configmap.yaml

# 4. Create Secrets (update values)
kubectl create secret generic shree-secrets \
  --from-literal=IBKR_HOST='127.0.0.1' \
  --from-literal=IBKR_PORT='4001' \
  --from-literal=IBKR_CLIENT_ID='1' \
  --from-literal=TELEGRAM_BOT_TOKEN='your-token-here' \
  --from-literal=TELEGRAM_CHAT_ID='your-chat-id'

# 5. Deploy application
kubectl apply -f deploy/k8s/deployment-with-secrets.yaml

# 6. Expose metrics service
kubectl apply -f deploy/prometheus/service.yaml

# 7. (Optional) Add ServiceMonitor for Prometheus Operator
kubectl apply -f deploy/prometheus/servicemonitor.yaml

# 8. Check status
kubectl get pods -l app=shree
kubectl logs -f deployment/shree
```

### Option 3: Quick Test (No Secrets)

```bash
# Deploy basic version without secrets
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/prometheus/service.yaml

# Port-forward and test
kubectl port-forward svc/shree-metrics 8000:8000 &
curl http://localhost:8000/metrics | grep shree
```

## 📊 Verify Prometheus Metrics

```bash
# Port-forward metrics endpoint
kubectl port-forward svc/shree-metrics 8000:8000

# In another terminal, query metrics
curl http://localhost:8000/metrics | grep shree_

# You should see:
# shree_live_bar_age_seconds{symbol="MES",timeframe="1m",env="prod"} 15.3
# shree_stale_episode_active{symbol="MES",env="prod"} 0
# shree_stale_live_bars_blocks_total{symbol="MES",env="prod"} 2
# shree_decisions_total{symbol="MES",env="prod",action="HOLD"} 45
# shree_pending_entry_orders_canceled_total{symbol="MES",env="prod",reason="STALE_LIVE_BARS"} 1
# shree_cancel_entries_calls_total{symbol="MES",env="prod",reason="STALE_LIVE_BARS",outcome="canceled"} 1

## 🧾 Trade closure audit trail (orders.db)

The bot persists deterministic trade closures to `data/orders.db` in the `trade_outcomes` table (keyed by `trade_cycle_id`).

### `exit_reason` semantics

`trade_outcomes.exit_reason` is written using the best available information at close time:

- **Explicit exits (preferred)**: when the system initiates an exit (e.g., `SIGNAL_EXIT`, `TIME_EXIT`, `STOP_LOSS`, `PROFIT_TARGET`, `TREND_CHANGE`), that reason is stored as a *pending exit reason* and then applied to `trade_outcomes` when the position transitions to flat.
- **Bracket fills**: if the position closes due to a bracket (TP/SL) without an explicit exit being initiated, the system attempts to infer **`PROFIT_TARGET` vs `STOP_LOSS`** by inspecting the order type of the last execution (`LIMIT`/`LMT` => TP, `STOP`/`STP`/`STOP_LIMIT` => SL). If inference is not possible, it falls back to `BRACKET_FILL`.
- **Backfilled history**: historical backfills use `BACKFILL` and should not be interpreted as the true execution reason.

This makes post-trade forensics queryable without relying on log retention.
```

## 🔧 Common PromQL Queries

### 1. Live Bar Age
```promql
shree_live_bar_age_seconds{timeframe="1m"}
```

### 2. Stale Episode Active
```promql
shree_stale_episode_active{symbol="MES"}
```

### 3. Stale Blocks Rate (Last 5min)
```promql
rate(shree_stale_live_bars_blocks_total[5m])
```

### 4. Canceled Entries Rate
```promql
rate(shree_pending_entry_orders_canceled_total{reason="STALE_LIVE_BARS"}[5m])
```

### 5. Decision Actions Distribution
```promql
sum by (action) (rate(shree_decisions_total[5m]))
```

### 6. Total Stale Blocks (Last Hour)
```promql
increase(shree_stale_live_bars_blocks_total[1h])
```

## 📈 Setup Grafana Dashboard

### If using Prometheus Operator (with Grafana)

```bash
# 1. Port-forward Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000

# 2. Open browser: http://localhost:3000
#    Default credentials: admin / prom-operator

# 3. Import dashboard
#    - Click '+' → Import
#    - Upload: deploy/grafana/dashboard.json
#    - Select Prometheus datasource
#    - Click Import
```

### Manual Prometheus Setup

```bash
# 1. Edit your prometheus.yml
# Add this scrape config:
#
# scrape_configs:
#   - job_name: 'shree'
#     static_configs:
#       - targets: ['shree-metrics.default.svc.cluster.local:8000']

# 2. Reload Prometheus
kubectl exec -n monitoring prometheus-0 -- kill -HUP 1

# 3. Verify target
kubectl port-forward -n monitoring svc/prometheus-k8s 9090:9090
# Visit http://localhost:9090/targets
# Look for 'shree' job
```

## 🔔 Setup Alerts

```bash
# Apply alert rules
kubectl apply -f deploy/prometheus/alerts.yml

# Verify alerts loaded
kubectl port-forward -n monitoring svc/prometheus-k8s 9090:9090
# Visit http://localhost:9090/alerts
```

## 🐛 Troubleshooting

### Pod not starting?
```bash
kubectl describe pod -l app=shree
kubectl logs -l app=shree --tail=50
```

### Metrics not appearing?
```bash
# Test from inside cluster
kubectl run curl --image=curlimages/curl -it --rm -- \
  curl http://shree-metrics:8000/metrics

# Check service
kubectl get svc shree-metrics
kubectl describe svc shree-metrics
```

### Image pull issues?
```bash
# For local testing with minikube
eval $(minikube docker-env)
docker build -t shree:latest .

# For local testing with kind
kind load docker-image shree:latest
```

## 📦 Push to Registry (Production)

### Docker Hub
```bash
docker tag shree:latest YOUR_USERNAME/shree:latest
docker push YOUR_USERNAME/shree:latest

# Update deployment.yaml:
# image: YOUR_USERNAME/shree:latest
```

### AWS ECR
```bash
# Login
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

# Create repo
aws ecr create-repository --repository-name shree

# Push
docker tag shree:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/shree:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/shree:latest

# Update deployment.yaml:
# image: YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/shree:latest
```

## 📚 Documentation

- **Complete Guide**: `docs/KUBERNETES_DEPLOYMENT.md`
- **Quick Reference**: `deploy/k8s/QUICKREF.md`
- **Observability Setup**: `docs/OBSERVABILITY.md`

## ✅ Deployment Checklist

- [ ] Docker image built
- [ ] Image tested locally
- [ ] ConfigMap applied
- [ ] Secrets created (with real credentials)
- [ ] Deployment applied
- [ ] Service created
- [ ] Pods running (check with `kubectl get pods`)
- [ ] Metrics endpoint accessible
- [ ] Prometheus scraping configured
- [ ] Grafana dashboard imported
- [ ] Alert rules applied
- [ ] Logs look healthy

## 🎯 Next Steps

1. **Monitor the deployment**: `kubectl logs -f deployment/shree`
2. **Check metrics**: Port-forward and curl /metrics
3. **Import Grafana dashboard**: Use `deploy/grafana/dashboard.json`
4. **Setup alerts**: Configure notification channels in AlertManager
5. **Production tuning**: Adjust resource limits based on actual usage

---

**Need help?** Check the full documentation in `docs/KUBERNETES_DEPLOYMENT.md`
