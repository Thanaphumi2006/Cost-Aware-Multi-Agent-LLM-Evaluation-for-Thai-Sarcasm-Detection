# Observability: load test -> Prometheus -> Grafana

Connect `loadtest.py` to Grafana to visualize cost/latency/routing.

## 1. Run the load test to emit Prometheus metrics

```powershell
cd Gold
C:\ve\Scripts\python.exe loadtest.py --rps 8 --duration 30 --concurrency 8 --prom metrics.prom
```

Produces `metrics.prom` in Prometheus textfile format:

```
sarcasm_requests_total 165
sarcasm_escalated_total 28
sarcasm_escalation_ratio 0.1697
sarcasm_cost_usd_total 0.029295
sarcasm_cost_usd_per_minute 0.087556
sarcasm_throughput_rps 8.221
sarcasm_latency_seconds{quantile="0.5"} 0.0281
sarcasm_route_latency_seconds{route="auto",quantile="0.5"} 0.0262
sarcasm_route_latency_seconds{route="escalated",quantile="0.5"} 0.8402
```

## 2. Have Prometheus collect this file

The easiest path is the **node_exporter textfile collector** (no need to write your own exporter):

```powershell
node_exporter.exe --collector.textfile.directory=C:\path\to\Gold
```

`prometheus.yml`:

```yaml
scrape_configs:
  - job_name: sarcasm
    scrape_interval: 10s
    static_configs:
      - targets: ["localhost:9100"]
```

> For graphs that move in real time, run `loadtest.py` repeatedly (e.g. every 30s)
> so it overwrites `metrics.prom`; the textfile collector reads the new values each time Prometheus scrapes.
> A single file that never updates = a flat line, which does not mean the system is idle.

## 3. Import the dashboard

Grafana -> Dashboards -> New -> Import -> upload `grafana_dashboard.json` -> pick the Prometheus datasource.

8 panels:

| panel | how to read it |
|---|---|
| Escalation ratio | fraction you must pay for; low = cheap (green <30%) |
| Cost $/min | cost at the current throughput |
| Throughput rps | far below target = concurrency is the bottleneck |
| Errors | should be 0 |
| Latency percentiles | p50 rides the free path (~26ms); the upper tail is escalated items |
| Latency by path | the auto vs escalated gap = **the price of uncertainty** (~32×) |
| Requests by path | stacked free vs paid ratio |
| Cumulative cost + calls | always paired (1 escalation = 1 call) |

## Caveats

- **`--mock` does not call the API** — latency is simulated from measured values (WCB 26ms / GPT 751ms / debate 4557ms).
  Use it to tune concurrency for free, unlimited times, then confirm once with `--live`.
- **n=127 is not production traffic** — this dashboard answers "is the orchestration correct / what's p95,"
  not "does it withstand 10k req/s." Don't claim beyond that.
- The **reportable F1 comes from `router.py`** (leave-fold-out), not from `loadtest.py`.
  loadtest uses a whole-set threshold because it measures the *system*, not the *accuracy*.
