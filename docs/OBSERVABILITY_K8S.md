# Kubernetes Observability for Shree (Bot runs locally)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Mac Mini (Local)                        │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Shree Bot (run_bot.py)                              │ │
│  │ - Runs as local process (launchd/terminal)             │ │
│  │ - Connects to IBKR TWS/Gateway                        │ │
│  │ - Executes trading logic                              │ │
│  │ - Exposes Prometheus metrics on 0.0.0.0:8000/metrics  │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│                            │ HTTP :8000/metrics              │
│                            ▼                                 │
└────────────────────────────┼─────────────────────────────────┘
                             │
                             │ Network (LAN or tunnel)
                             │
┌────────────────────────────▼─────────────────────────────────┐
│              Kubernetes Cluster (Observability)              │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Prometheus                                             │ │
│  │ - Scrapes Mac mini IP:8000/metrics every 30s          │ │
│  │ - Stores time-series data                             │ │
│  │ - Evaluates alert rules                               │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│  ┌────────────────────────▼────────────────────────────────┐ │
│  │ Grafana (optional)                                     │ │
│  │ - Visualizes metrics dashboards                       │ │
│  │ - Shows trading performance, staleness, cancellations │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Alertmanager (optional)                                │ │
│  │ - Routes alerts to Slack/PagerDuty/Email              │ │
│  │ - Manages alert deduplication and silencing          │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

**Key Points:**
- ✅ Bot runs **locally** on Mac mini (no containerization)
- ✅ Kubernetes used **only** for monitoring stack
- ✅ Prometheus scrapes bot metrics over network
- ✅ No changes to bot startup or execution model
- ✅ Existing launchd/terminal workflows preserved

---

## Part 1: Configure Bot on Mac Mini

### 1.1 Set Environment Variables

The bot needs these environment variables to expose Prometheus metrics:

```bash
# Add to your shell profile (~/.zshrc or ~/.bash_profile)
export PROMETHEUS_ENABLED=true
export PROMETHEUS_ADDR=0.0.0.0  # Bind to all interfaces (required for k8s scraping)
export PROMETHEUS_PORT=8000
export DEPLOY_ENV=prod  # or 'stage', 'dev' - used as metric label
```

**Important:** `PROMETHEUS_ADDR=0.0.0.0` is required so Kubernetes can reach the metrics endpoint from outside the Mac mini.

### 1.2 Restart the Bot

After setting environment variables, restart your bot:

```bash
# If using launchd
launchctl stop com.shree.bot  # adjust service name
launchctl start com.shree.bot

# If running manually
# Stop existing process (Ctrl+C or kill)
python run_bot.py  # or however you start it
```

### 1.3 Verify Metrics Endpoint Locally

Test that metrics are exposed correctly:

```bash
# From Mac mini itself
curl http://localhost:8000/metrics

# You should see Prometheus metrics like:
# # HELP shree_live_bar_age_seconds Age of the most recent live bar
# shree_live_bar_age_seconds{env="prod",symbol="MES",timeframe="1m"} 2.5
# shree_stale_episode_active{env="prod",symbol="MES",timeframe="1m"} 0.0
# ...
```

Run the validation script:

```bash
./scripts/check_metrics_reachable.sh localhost 8000
```

Expected output:
```
✅ All checks passed! Metrics endpoint is healthy.
```

### 1.4 Get Your Mac Mini IP Address

Kubernetes needs to know your Mac mini's IP address on the local network:

```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

Example output:
```
inet 192.168.1.100 netmask 0xffffff00 broadcast 192.168.1.255
```

**Note:** Use the `192.168.x.x` or `10.x.x.x` address (local network IP), not `127.0.0.1`.

### 1.5 Firewall Configuration

Ensure your Mac firewall allows connections to port 8000:

**Option A: Allow all incoming (less secure)**
```bash
# System Preferences → Security & Privacy → Firewall
# Turn off firewall OR add exception for Python
```

**Option B: Allow specific port (recommended)**
```bash
# Allow incoming on port 8000 from local network only
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /path/to/python
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /path/to/python
```

**Option C: Use pfctl for precise control**
```bash
# Add rule to allow port 8000 from k8s cluster subnet
# (Advanced - consult macOS pfctl documentation)
```

### 1.6 Test from Kubernetes Cluster

Once your k8s cluster is set up, test connectivity from inside a pod:

```bash
# Run a test pod in k8s
kubectl run test-curl --rm -it --image=curlimages/curl --restart=Never -- \
  curl -v http://192.168.1.100:8000/metrics

# Or if you already have a pod running:
kubectl exec -it <any-pod> -- curl http://192.168.1.100:8000/metrics
```

If this fails, you have a network connectivity issue (firewall, routing, etc.).

---

## Part 2: Configure Kubernetes Prometheus

### 2.1 Prerequisites

You need one of these setups:

**Option A: Prometheus Operator (Recommended)**
- Install with Helm:
  ```bash
  helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
  helm repo update
  helm install prometheus prometheus-community/kube-prometheus-stack \
    --namespace monitoring --create-namespace
  ```

**Option B: Standalone Prometheus**
- Deploy Prometheus manually
- Will need to manually edit ConfigMap for scrape configs

### 2.2 Get Your Mac Mini IP

Update the scrape configuration with your actual Mac mini IP:

```bash
# Find your IP again
ifconfig | grep "inet " | grep -v 127.0.0.1

# Example: 192.168.1.100
```

Edit `deploy/k8s-observability/prometheus-scrape-config.yaml`:

```yaml
# Line 17: Update this IP address
- ip: 192.168.1.100  # <-- CHANGE TO YOUR MAC MINI IP
```

### 2.3 Apply Kubernetes Manifests

**If using Prometheus Operator:**

```bash
# Apply external endpoint + ServiceMonitor
kubectl apply -f deploy/k8s-observability/prometheus-scrape-config.yaml

# Apply alert rules
kubectl apply -f deploy/k8s-observability/prometheus-rules.yaml
```

**If using standalone Prometheus:**

You'll need to add a scrape config to your Prometheus ConfigMap:

```bash
# Get your current Prometheus config
kubectl get configmap prometheus-config -n monitoring -o yaml > /tmp/prometheus-config.yaml

# Edit it and add this scrape_config:
#
# scrape_configs:
#   - job_name: 'shree-mac-mini'
#     static_configs:
#       - targets:
#           - '192.168.1.100:8000'  # Your Mac mini IP
#         labels:
#           instance: 'mac-mini'
#           env: 'prod'
#     scrape_interval: 30s
#     scrape_timeout: 10s
#     metrics_path: '/metrics'

# Apply the updated config
kubectl apply -f /tmp/prometheus-config.yaml

# Reload Prometheus
kubectl rollout restart deployment prometheus -n monitoring
```

### 2.4 Verify Scraping Works

**Check Prometheus Targets:**

1. Port-forward to Prometheus UI:
   ```bash
   kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
   ```

2. Open browser: http://localhost:9090/targets

3. Look for `shree-external` or `shree-mac-mini` target

4. Verify status is **UP** (green)

**Query Metrics:**

In Prometheus UI (http://localhost:9090/graph):

```promql
# Check if metrics are being scraped
up{job="shree-mac-mini"}

# View bar age
shree_live_bar_age_seconds

# View stale blocks
shree_stale_live_bars_blocks_total

# Decision rate
rate(shree_decisions_total[5m])
```

### 2.5 Verify Alerts

Check that alert rules loaded:

```bash
# Port-forward if needed
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090

# Open browser: http://localhost:9090/alerts
# You should see:
#   - ShreeStaleBarsCritical
#   - ShreeFrequentStaleBlocks
#   - ShreeFrequentStaleCancellations
#   - ShreeMetricsDown
#   - etc.
```

---

## Part 3: Network Connectivity Options

### Option A: Direct IP (Simple, LAN-only)

✅ **Best if:** Your k8s cluster is on the same LAN as your Mac mini (e.g., Docker Desktop k8s on same machine, or local k3s cluster)

**Setup:**
- Use Mac mini's local IP (192.168.x.x or 10.x.x.x)
- Ensure no firewall blocking
- Done! ✅

**Pros:**
- Simplest setup
- No additional tools

**Cons:**
- Only works on same network
- Mac IP must be stable (consider DHCP reservation)

---

### Option B: Tailscale VPN (Recommended for remote clusters)

✅ **Best if:** Your k8s cluster is remote (cloud, different network) or you want secure encrypted tunnel

**Setup:**

1. Install Tailscale on Mac mini:
   ```bash
   brew install tailscale
   sudo tailscale up
   ```

2. Get your Tailscale IP:
   ```bash
   tailscale ip -4
   # Example output: 100.101.102.103
   ```

3. Install Tailscale in Kubernetes cluster:
   ```bash
   # Deploy Tailscale subnet router in k8s
   kubectl apply -f https://raw.githubusercontent.com/tailscale/tailscale/main/docs/k8s/proxy.yaml
   ```

4. Update scrape config with Tailscale IP:
   ```yaml
   - ip: 100.101.102.103  # Your Tailscale IP
   ```

**Pros:**
- Works anywhere (LAN, WAN, cloud)
- Encrypted tunnel
- NAT traversal (no port forwarding needed)
- Static IPs

**Cons:**
- Requires Tailscale account (free for personal use)
- Additional dependency

---

### Option C: SSH Reverse Tunnel

✅ **Best if:** You can SSH to a machine reachable from k8s (e.g., jump host, cloud VM)

**Setup:**

1. From Mac mini, create reverse tunnel to accessible host:
   ```bash
   ssh -R 8000:localhost:8000 user@accessible-host
   # Keep this connection alive (use screen/tmux or systemd service)
   ```

2. Update scrape config to point to accessible-host:
   ```yaml
   - ip: <accessible-host-ip>
     ports:
       - name: metrics
         port: 8000  # Tunneled from Mac mini
   ```

**Pros:**
- No additional tools if SSH already available
- Works through NAT/firewalls

**Cons:**
- Must maintain persistent SSH connection
- Less reliable (can disconnect)
- Not encrypted beyond SSH tunnel

---

### Option D: Cloudflare Tunnel

✅ **Best if:** You want public metrics endpoint or remote access without VPN

**Setup:**

1. Install Cloudflare Tunnel:
   ```bash
   brew install cloudflare/cloudflare/cloudflared
   cloudflared tunnel login
   cloudflared tunnel create shree
   ```

2. Create tunnel config:
   ```yaml
   # ~/.cloudflared/config.yml
   tunnel: <tunnel-id>
   credentials-file: /path/to/credentials.json
   ingress:
     - hostname: shree-metrics.example.com
       service: http://localhost:8000
     - service: http_status:404
   ```

3. Run tunnel:
   ```bash
   cloudflared tunnel run shree
   ```

4. Update Prometheus to scrape HTTPS endpoint:
   ```yaml
   - job_name: 'shree-cloudflare'
     static_configs:
       - targets:
           - shree-metrics.example.com:443
     scheme: https
   ```

**Pros:**
- Works from anywhere
- HTTPS/encrypted
- Built-in DDoS protection
- Free for personal use

**Cons:**
- Metrics exposed to internet (add auth or IP restrictions)
- Requires Cloudflare account
- DNS setup required

---

## Part 4: Validation Checklist

Run through this checklist to ensure everything works:

### ✅ Local Bot Checks

- [ ] `PROMETHEUS_ENABLED=true` set
- [ ] `PROMETHEUS_ADDR=0.0.0.0` set
- [ ] `PROMETHEUS_PORT=8000` set
- [ ] `DEPLOY_ENV=prod` set (or your preferred env)
- [ ] Bot is running: `ps aux | grep run_bot.py`
- [ ] Local curl works: `curl http://localhost:8000/metrics`
- [ ] Validation script passes: `./scripts/check_metrics_reachable.sh`
- [ ] Metrics include `shree_live_bar_age_seconds`, `shree_stale_episode_active`, etc.

### ✅ Network Connectivity Checks

- [ ] Mac mini IP obtained: `ifconfig | grep inet`
- [ ] Firewall allows port 8000
- [ ] Curl from k8s pod works: `kubectl run test-curl ...`
- [ ] (If Tailscale) Tailscale IP obtained and k8s can reach it
- [ ] (If SSH tunnel) Tunnel is active and stable
- [ ] (If Cloudflare) Domain resolves and HTTPS works

### ✅ Kubernetes Checks

- [ ] Prometheus is deployed (Operator or standalone)
- [ ] Scrape config updated with correct Mac mini IP
- [ ] `kubectl apply -f deploy/k8s-observability/` succeeded
- [ ] Endpoints created: `kubectl get endpoints shree-external`
- [ ] Service created: `kubectl get svc shree-external`
- [ ] ServiceMonitor created: `kubectl get servicemonitor shree-external`
- [ ] PrometheusRule created: `kubectl get prometheusrule shree-alerts`

### ✅ Prometheus Scraping Checks

- [ ] Prometheus UI accessible (port-forward 9090)
- [ ] Targets page shows `shree-external` or `shree-mac-mini`
- [ ] Target status is **UP** (not DOWN)
- [ ] Last scrape shows recent timestamp
- [ ] Query `up{job="shree-mac-mini"}` returns `1`
- [ ] Query `shree_live_bar_age_seconds` returns data
- [ ] Graph shows time-series data accumulating

### ✅ Alert Rules Checks

- [ ] Alerts page shows Shree alert rules
- [ ] Rules are in **Inactive** state (green - no firing alerts)
- [ ] Simulate stale condition (stop bot or disconnect IBKR)
- [ ] After 5m, `ShreeStaleBarsCritical` should fire
- [ ] After 2m, `ShreeMetricsDown` should fire (if bot stopped)

### ✅ Optional: Grafana Dashboard

- [ ] Grafana deployed and accessible
- [ ] Prometheus added as data source
- [ ] Import dashboard from `deploy/grafana/dashboard.json`
- [ ] Panels show live data
- [ ] Bar age graph displays correctly
- [ ] Decision rate panels populate

---

## Part 5: Troubleshooting

### Problem: Prometheus target shows DOWN

**Symptoms:**
- Prometheus UI shows target as red/DOWN
- Error: "Connection refused" or "Timeout"

**Solutions:**

1. **Check bot is running:**
   ```bash
   ps aux | grep run_bot.py
   # If not running, start it
   ```

2. **Check metrics endpoint locally:**
   ```bash
   curl http://localhost:8000/metrics
   # If this fails, bot metrics server isn't starting
   # Check PROMETHEUS_ENABLED=true
   ```

3. **Check firewall:**
   ```bash
   # Test from another machine on same network
   curl http://<mac-mini-ip>:8000/metrics
   # If this fails, firewall is blocking
   ```

4. **Check k8s can reach Mac mini:**
   ```bash
   kubectl run test-curl --rm -it --image=curlimages/curl --restart=Never -- \
     curl -v http://<mac-mini-ip>:8000/metrics
   # If this fails, network routing issue
   ```

5. **Verify IP address is correct:**
   ```bash
   # On Mac mini
   ifconfig | grep "inet "
   # Compare to deploy/k8s-observability/prometheus-scrape-config.yaml
   ```

---

### Problem: Metrics are stale/not updating

**Symptoms:**
- Prometheus shows old data
- Last scrape timestamp is old
- Graph flatlines

**Solutions:**

1. **Check scrape interval:**
   ```yaml
   # Should be 30s or similar, not too long
   interval: 30s
   ```

2. **Check Prometheus hasn't crashed:**
   ```bash
   kubectl get pods -n monitoring
   kubectl logs -n monitoring prometheus-xxx
   ```

3. **Check bot is actively updating metrics:**
   ```bash
   # Curl metrics twice with delay
   curl http://localhost:8000/metrics | grep shree_live_bar_age_seconds
   sleep 5
   curl http://localhost:8000/metrics | grep shree_live_bar_age_seconds
   # Values should change
   ```

---

### Problem: Alerts not firing

**Symptoms:**
- PrometheusRule applied successfully
- Conditions met (e.g., bars are stale)
- But alert doesn't fire in Prometheus UI

**Solutions:**

1. **Check alert rules loaded:**
   ```bash
   kubectl get prometheusrule -n monitoring
   # Should show shree-alerts
   ```

2. **Check Prometheus picked up rules:**
   ```bash
   # Prometheus UI → Alerts
   # Should see Shree* alerts listed
   ```

3. **Check alert evaluation:**
   ```bash
   # Run the alert query manually:
   shree_live_bar_age_seconds{timeframe="1m"} > 120
   # If this returns results, alert should fire after 'for' duration
   ```

4. **Check 'for' duration:**
   ```yaml
   for: 5m  # Alert must be true for 5 minutes before firing
   ```

5. **Check label selectors match:**
   ```yaml
   # If using Prometheus Operator, check:
   ruleSelector:
     matchLabels:
       release: prometheus  # Must match your PrometheusRule labels
   ```

---

### Problem: Mac mini IP changes

**Symptoms:**
- Scraping was working, now DOWN
- DHCP renewed IP address

**Solutions:**

1. **Set static IP or DHCP reservation:**
   ```bash
   # In router settings, reserve IP for Mac mini's MAC address
   # Or set static IP in System Preferences → Network
   ```

2. **Use Tailscale (stable IPs):**
   ```bash
   # Tailscale IPs don't change
   tailscale ip -4
   # Use this in scrape config instead
   ```

3. **Update scrape config if IP changed:**
   ```bash
   # Edit deploy/k8s-observability/prometheus-scrape-config.yaml
   # Update IP, then:
   kubectl apply -f deploy/k8s-observability/prometheus-scrape-config.yaml
   ```

---

## Part 6: Key PromQL Queries

Use these queries in Prometheus or Grafana:

### Staleness Monitoring

```promql
# Current bar age
shree_live_bar_age_seconds{timeframe="1m"}

# Bar age over time (graph)
shree_live_bar_age_seconds{symbol="MES", timeframe="1m"}

# Stale episode active (0 or 1)
shree_stale_episode_active

# Stale blocks rate (per minute)
rate(shree_stale_live_bars_blocks_total[5m]) * 60

# Total stale blocks today
increase(shree_stale_live_bars_blocks_total[24h])
```

### Decision Monitoring

```promql
# Decision rate (decisions per second)
rate(shree_decisions_total[5m])

# Decisions by action type
sum by (action) (shree_decisions_total)

# HOLD vs BUY/SELL ratio
sum(shree_decisions_total{action="HOLD"}) /
sum(shree_decisions_total{action=~"BUY|SELL"})
```

### Cancellation Monitoring

```promql
# Cancellation rate (per minute)
rate(shree_pending_entry_orders_canceled_total[5m]) * 60

# Cancellations by reason
sum by (reason) (shree_pending_entry_orders_canceled_total)

# Stale bar cancellations (24h)
increase(shree_pending_entry_orders_canceled_total{reason="STALE_LIVE_BARS"}[24h])

# Cancel call outcomes
sum by (outcome) (shree_cancel_entries_calls_total)
```

### Health Monitoring

```promql
# Bot is up (1 = up, 0 = down)
up{job="shree-mac-mini"}

# Time since last successful scrape
time() - timestamp(up{job="shree-mac-mini"})

# Scrape duration
scrape_duration_seconds{job="shree-mac-mini"}
```

---

## Part 7: Advanced: Secure Public Metrics

If you need to expose metrics over the internet (e.g., cloud k8s cluster), add authentication:

### Option 1: Basic Auth with Nginx Reverse Proxy

```bash
# Install nginx on Mac mini
brew install nginx

# Create htpasswd file
htpasswd -c /usr/local/etc/nginx/.htpasswd prometheus

# Configure nginx
cat > /usr/local/etc/nginx/nginx.conf <<EOF
events {}
http {
  server {
    listen 8001;
    location /metrics {
      auth_basic "Prometheus Metrics";
      auth_basic_user_file /usr/local/etc/nginx/.htpasswd;
      proxy_pass http://localhost:8000/metrics;
    }
  }
}
EOF

# Start nginx
brew services start nginx

# Update Prometheus scrape config
# - target: mac-mini:8001
# - basic_auth:
#     username: prometheus
#     password: <from htpasswd>
```

### Option 2: mTLS (Mutual TLS)

Use Prometheus's TLS config with client certificates (advanced, see Prometheus docs).

### Option 3: VPN Only (Recommended)

Use Tailscale/WireGuard and keep metrics on private VPN network only.

---

## Summary

**What you did:**
1. ✅ Configured bot to expose Prometheus metrics on `0.0.0.0:8000/metrics`
2. ✅ Bot continues running locally (no k8s deployment)
3. ✅ Created k8s manifests for Prometheus to scrape Mac mini
4. ✅ Created alert rules for stale bars, cancellations, and health
5. ✅ Validated end-to-end: bot → network → Prometheus → alerts

**What you didn't do:**
- ❌ Containerize the bot
- ❌ Deploy bot to Kubernetes
- ❌ Change bot startup model
- ❌ Add k8s dependencies to bot code

**Next steps:**
- Import Grafana dashboard (optional)
- Set up Alertmanager for notifications (optional)
- Add more custom alerts as needed
- Monitor and iterate!

---

## Quick Reference

| Task | Command |
|------|---------|
| Check bot running | `ps aux \| grep run_bot.py` |
| Test metrics locally | `curl http://localhost:8000/metrics` |
| Get Mac mini IP | `ifconfig \| grep "inet " \| grep -v 127.0.0.1` |
| Validate metrics | `./scripts/check_metrics_reachable.sh` |
| Apply k8s manifests | `kubectl apply -f deploy/k8s-observability/` |
| Port-forward Prometheus | `kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090` |
| Check Prometheus targets | Open http://localhost:9090/targets |
| Check alerts | Open http://localhost:9090/alerts |
| Test from k8s | `kubectl run test-curl --rm -it --image=curlimages/curl --restart=Never -- curl http://<MAC-IP>:8000/metrics` |
