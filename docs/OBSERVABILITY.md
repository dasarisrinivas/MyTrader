Observability: Prometheus metrics for MyTrader

Overview
--------
This project exposes optional Prometheus metrics. The exporter is opt-in; enable it via environment variables or your config.

Important note on metric path
-----------------------------
The built-in helper uses prometheus_client.start_http_server(), which always serves metrics on /metrics. You cannot change the path with PROMETHEUS_PATH. If you need a custom path (e.g., /monitor/metrics), expose the CollectorRegistry via your FastAPI/uvicorn app instead.

Environment
-----------
- PROMETHEUS_ENABLED=true
- PROMETHEUS_ADDR=0.0.0.0
- PROMETHEUS_PORT=8000
- DEPLOY_ENV=prod

Behavior
--------
- SignalProcessor emits staleness-related metrics:
  - mytrader_live_bar_age_seconds{symbol,timeframe,env}
  - mytrader_stale_live_bars_blocks_total{symbol,env}
  - mytrader_stale_episode_active{symbol,env}
  - mytrader_decisions_total{symbol,env,action}

- TradeExecutor emits cancellation metrics (single source of truth for cancels):
  - mytrader_pending_entry_orders_canceled_total{symbol,env,reason}
  - mytrader_cancel_entries_calls_total{symbol,env,reason,outcome}

Deployment (Kubernetes)
-----------------------
Example manifests are provided:
- `deploy/k8s/deployment.yaml` — Pod with app: mytrader label and containerPort 8000
- `deploy/prometheus/service.yaml` — Service exposing metrics port
- `deploy/prometheus/servicemonitor.yaml` — ServiceMonitor for Prometheus Operator

Basic deployment steps:
```bash
# Build your Docker image
docker build -t mytrader:latest .

# Apply k8s manifests
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/prometheus/service.yaml
kubectl apply -f deploy/prometheus/servicemonitor.yaml

# Verify pod is running and metrics port exposed
kubectl get pods -l app=mytrader
kubectl port-forward svc/mytrader-metrics 8000:8000
curl http://localhost:8000/metrics
```

Prometheus Operator (ServiceMonitor) example
-------------------------------------------
Make sure your pod has label `app: mytrader` and exposes containerPort 8000. The provided Service and ServiceMonitor will allow Prometheus Operator to discover the metrics endpoint.

If you don't use Prometheus Operator, manually add a Prometheus scrape config:
```yaml
scrape_configs:
  - job_name: 'mytrader'
    static_configs:
      - targets: ['mytrader-metrics.default.svc.cluster.local:8000']
```

Alert rules
-----------
See `deploy/prometheus/alerts.yml` for starter alerting rules. They rely on the metrics described above.

Notes
-----
- We intentionally avoid high-cardinality labels. Use symbol/timeframe/env only.
- If you want a custom /metrics path or to integrate with an existing FastAPI app, I can add a small FastAPI endpoint that returns the Registry metrics (recommended for path control and integration with existing HTTP server).
