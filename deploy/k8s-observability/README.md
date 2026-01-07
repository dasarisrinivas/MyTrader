# Kubernetes Observability Deployment - Quick Start

## What This Does

Deploys **Prometheus monitoring stack** to Kubernetes to scrape metrics from your MyTrader bot running **locally** on Mac mini.

**Important:** The bot itself does NOT run in Kubernetes. Only monitoring tools run in k8s.

---

## Prerequisites

- ✅ Kubernetes cluster running (Docker Desktop, minikube, k3s, or cloud)
- ✅ `kubectl` configured and working
- ✅ MyTrader bot running locally on Mac mini
- ✅ Bot has environment variables set:
  ```bash
  export PROMETHEUS_ENABLED=true
  export PROMETHEUS_ADDR=0.0.0.0
  export PROMETHEUS_PORT=8000
  export DEPLOY_ENV=prod
  ```

---

## Step 1: Verify Bot Metrics Locally

Before deploying to k8s, ensure metrics work locally:

```bash
# From Mac mini terminal
curl http://localhost:8000/metrics

# Or use the validation script
./scripts/check_metrics_reachable.sh localhost 8000
```

Expected output should include lines like:
```
# HELP mytrader_live_bar_age_seconds Age of the most recent live bar
mytrader_live_bar_age_seconds{env="prod",symbol="MES",timeframe="1m"} 3.2
mytrader_stale_episode_active{env="prod",symbol="MES",timeframe="1m"} 0.0
...
```

If this fails, the bot is not exposing metrics. Check:
- Bot is running: `ps aux | grep run_bot.py`
- Environment variables are set (see above)
- Restart bot after setting env vars

---

## Step 2: Install Prometheus in Kubernetes

### Option A: Using Prometheus Operator (Recommended)

```bash
# Add Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install kube-prometheus-stack (includes Prometheus, Grafana, Alertmanager)
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=prometheus -n monitoring --timeout=300s
```

### Option B: Standalone Prometheus (Manual)

If you prefer not to use Helm, deploy Prometheus manually:

```bash
# Create namespace
kubectl create namespace monitoring

# Deploy Prometheus (use official manifests or your own)
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/example/prometheus-operator-crd/monitoring.coreos.com_prometheuses.yaml
# ... (additional setup required - see Prometheus docs)
```

---

## Step 3: Deploy MyTrader Scrape Configuration

### Update IP Address (if needed)

The scrape config is pre-filled with your Mac mini IP: `192.168.4.157`

If your IP changed, update it:

```bash
# Get current IP
ifconfig | grep "inet " | grep -v 127.0.0.1

# Edit the file
nano deploy/k8s-observability/prometheus-scrape-config.yaml

# Line 17: Update IP address
- ip: 192.168.4.157  # <-- Update this if needed
```

### Apply Manifests

```bash
# Apply external endpoint + ServiceMonitor
kubectl apply -f deploy/k8s-observability/prometheus-scrape-config.yaml

# Apply alert rules
kubectl apply -f deploy/k8s-observability/prometheus-rules.yaml
```

Expected output:
```
endpoints/mytrader-external created
service/mytrader-external created
servicemonitor.monitoring.coreos.com/mytrader-external created
prometheusrule.monitoring.coreos.com/mytrader-alerts created
```

Verify:
```bash
kubectl get endpoints mytrader-external
kubectl get svc mytrader-external
kubectl get servicemonitor mytrader-external
kubectl get prometheusrule mytrader-alerts
```

---

## Step 4: Verify Prometheus is Scraping

### Test Network Connectivity

First, verify k8s can reach your Mac mini:

```bash
# Run a test curl from within the cluster
kubectl run test-curl --rm -it --image=curlimages/curl --restart=Never -- \
  curl -v http://192.168.4.157:8000/metrics
```

If this **fails**, you have a network issue:
- Firewall blocking port 8000
- Mac mini IP changed
- Network routing issue between k8s and Mac mini

If it **succeeds**, Prometheus should be able to scrape.

### Check Prometheus Targets

```bash
# Port-forward Prometheus UI
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090

# Open browser: http://localhost:9090

# Navigate to: Status → Targets

# Look for: "mytrader-external" or "serviceMonitor/default/mytrader-external/0"
# Status should be: UP (green)
```

If status is **DOWN**:
- Check network connectivity (test-curl above)
- Check bot is running and metrics exposed
- Check IP address is correct in scrape config

### Query Metrics

In Prometheus UI, go to Graph tab and try these queries:

```promql
# Check scrape is working
up{job=~".*mytrader.*"}

# View bar age
mytrader_live_bar_age_seconds

# View stale blocks
rate(mytrader_stale_live_bars_blocks_total[5m])

# Decision distribution
sum by (action) (mytrader_decisions_total)
```

If metrics don't appear:
- Wait 30-60 seconds (scrape interval)
- Check target is UP
- Check bot is actively trading (metrics only emitted when bot is active)

---

## Step 5: Verify Alerts

```bash
# Open Prometheus UI (port-forward if needed)
# Navigate to: Alerts

# You should see these rules:
# - MyTraderStaleBarsCritical
# - MyTraderStaleBarsWarning
# - MyTraderFrequentStaleBlocks
# - MyTraderStaleEpisodeLong
# - MyTraderFrequentStaleCancellations
# - MyTraderHighCancellationRate
# - MyTraderMetricsDown
# - MyTraderNoRecentDecisions
```

All should be in **Inactive** (green) state under normal conditions.

### Test Alert Firing

To test alerts work:

1. **Stop the bot** (simulates metrics down):
   ```bash
   # On Mac mini, stop the bot
   # Wait 2 minutes
   # Check Prometheus → Alerts
   # "MyTraderMetricsDown" should fire
   ```

2. **Restart bot** - alert should clear within a scrape cycle

---

## Step 6: Access Grafana (Optional)

If you installed kube-prometheus-stack, Grafana is included:

```bash
# Get Grafana admin password
kubectl get secret -n monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 --decode
echo

# Port-forward Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Open browser: http://localhost:3000
# Username: admin
# Password: (from above command)
```

### Import MyTrader Dashboard

1. In Grafana, go to Dashboards → Import
2. Upload `deploy/grafana/dashboard.json`
3. Select Prometheus data source
4. Click Import

You should see panels for:
- Live bar age (graph)
- Stale episode active (stat)
- Decision rate (graph)
- Stale blocks (graph)
- Canceled entries (graph)
- Total stale blocks (stat)
- Cancellation outcomes (pie chart)

---

## Troubleshooting

### Problem: Target shows DOWN in Prometheus

**Check 1: Bot is running**
```bash
ps aux | grep run_bot.py
```

**Check 2: Metrics endpoint works locally**
```bash
curl http://localhost:8000/metrics
```

**Check 3: Firewall allows connections**
```bash
# Test from k8s
kubectl run test-curl --rm -it --image=curlimages/curl --restart=Never -- \
  curl -v http://192.168.4.157:8000/metrics
```

**Check 4: IP address is correct**
```bash
# On Mac mini
ifconfig | grep "inet " | grep -v 127.0.0.1

# Compare to deploy/k8s-observability/prometheus-scrape-config.yaml
```

---

### Problem: Metrics are all zeros or missing

**Cause:** Bot may not be actively trading yet, so metrics haven't been emitted.

**Solution:** Wait for bot to receive live bars and make trading decisions. Metrics are emitted when events occur (bar received, decision made, etc.).

**Verify bot is active:**
```bash
# Check bot logs
tail -f /path/to/bot/logs/mytrader.log

# Look for:
# - "Live bar received"
# - "Decision: BUY/SELL/HOLD"
# - "Prometheus metrics updated"
```

---

### Problem: Alerts not firing when they should

**Check 1: Alert rules loaded**
```bash
kubectl get prometheusrule mytrader-alerts -o yaml
```

**Check 2: Prometheus picked up rules**
```bash
# Prometheus UI → Alerts
# Rules should be listed (even if Inactive)
```

**Check 3: Alert query works**
```bash
# In Prometheus UI, run the alert query manually:
mytrader_live_bar_age_seconds{timeframe="1m"} > 120

# If this returns results, alert should fire after 'for' duration
```

**Check 4: Label selectors match**
```bash
# If using Prometheus Operator, ensure ruleSelector matches
kubectl get prometheus -n monitoring -o yaml | grep -A5 ruleSelector
```

---

## Summary of What Was Deployed

### In Kubernetes:
- ✅ Prometheus (if not already present)
- ✅ External Endpoints pointing to Mac mini IP
- ✅ Service for external endpoints
- ✅ ServiceMonitor to configure scraping
- ✅ PrometheusRule with alert definitions
- ✅ (Optional) Grafana dashboard

### On Mac Mini:
- ✅ MyTrader bot runs locally (unchanged)
- ✅ Prometheus metrics exposed on `:8000/metrics`
- ✅ No containers, no k8s deployments for the bot

### Network Flow:
```
Mac Mini (192.168.4.157:8000)
    ↓
    ↓ HTTP /metrics scrape every 30s
    ↓
Kubernetes Prometheus
    ↓
    ↓ Time-series storage
    ↓
Grafana (visualization) + Alertmanager (notifications)
```

---

## Next Steps

1. **Monitor for a day** - Let Prometheus collect data and verify stability

2. **Set up Alertmanager** (optional) - Route alerts to Slack/PagerDuty:
   ```bash
   # Already installed with kube-prometheus-stack
   # Configure in Alertmanager ConfigMap
   kubectl edit configmap -n monitoring alertmanager-prometheus-kube-prometheus-alertmanager
   ```

3. **Create custom dashboards** - Add more Grafana panels as needed

4. **Add more alerts** - Customize alert rules in `prometheus-rules.yaml`

5. **Secure metrics endpoint** - If exposing over internet, add auth (see docs/OBSERVABILITY_K8S.md Part 7)

6. **Set up remote storage** (optional) - For long-term metrics retention:
   - Thanos
   - Cortex
   - Victoria Metrics

---

## Cleanup (if needed)

To remove everything:

```bash
# Delete MyTrader monitoring resources
kubectl delete -f deploy/k8s-observability/

# Delete Prometheus stack (if installed via Helm)
helm uninstall prometheus -n monitoring

# Delete namespace
kubectl delete namespace monitoring
```

Bot on Mac mini continues running unaffected.

---

## Files Created

- `deploy/k8s-observability/prometheus-scrape-config.yaml` - External endpoint + ServiceMonitor
- `deploy/k8s-observability/prometheus-rules.yaml` - Alert rules
- `scripts/check_metrics_reachable.sh` - Validation script
- `docs/OBSERVABILITY_K8S.md` - Comprehensive documentation

## Files Modified

- None (bot code unchanged)

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `curl http://localhost:8000/metrics` | Test bot metrics locally |
| `kubectl apply -f deploy/k8s-observability/` | Deploy monitoring config |
| `kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090` | Access Prometheus UI |
| `kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80` | Access Grafana UI |
| `kubectl get prometheusrule` | List alert rules |
| `kubectl logs -n monitoring prometheus-xxx` | Check Prometheus logs |
| `./scripts/check_metrics_reachable.sh` | Validate metrics endpoint |

---

**Your Mac mini IP:** `192.168.4.157`  
**Metrics endpoint:** `http://192.168.4.157:8000/metrics`  
**Scrape interval:** `30s`  
**Environment:** `prod` (from DEPLOY_ENV)
