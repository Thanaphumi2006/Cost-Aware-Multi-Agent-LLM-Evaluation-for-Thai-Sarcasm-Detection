# Observability — load test -> Prometheus -> Grafana

ต่อ `loadtest.py` เข้ากับ Grafana เพื่อดู cost/latency/routing แบบเห็นภาพ

## 1. รัน load test ให้ออก Prometheus metrics

```powershell
cd Gold
C:\ve\Scripts\python.exe loadtest.py --rps 8 --duration 30 --concurrency 8 --prom metrics.prom
```

ได้ไฟล์ `metrics.prom` รูปแบบ Prometheus textfile:

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

## 2. ให้ Prometheus เก็บไฟล์นี้

ทางที่ง่ายที่สุดคือ **node_exporter textfile collector** (ไม่ต้องเขียน exporter เอง):

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

> ถ้าอยากได้กราฟที่เดินตามเวลาจริง ต้องรัน `loadtest.py` ซ้ำเป็นรอบๆ (เช่นทุก 30 วิ)
> ให้มันเขียนทับ `metrics.prom` — textfile collector จะอ่านค่าใหม่ทุกครั้งที่ Prometheus scrape
> ไฟล์เดียวที่ไม่อัปเดต = กราฟเส้นแบน ไม่ได้แปลว่าระบบนิ่ง

## 3. import dashboard

Grafana -> Dashboards -> New -> Import -> อัปโหลด `grafana_dashboard.json` -> เลือก Prometheus datasource

8 panel:

| panel | อ่านยังไง |
|---|---|
| Escalation ratio | สัดส่วนที่ต้องจ่ายเงิน — ต่ำ = ประหยัด (เขียว <30%) |
| Cost $/นาที | ต้นทุนที่ throughput ปัจจุบัน |
| Throughput rps | ถ้าต่ำกว่าเป้ามาก = concurrency เป็นคอขวด |
| Errors | ควรเป็น 0 |
| Latency percentiles | p50 เกาะเส้นทางฟรี (~26ms) หางบนคือข้อที่ escalate |
| Latency แยกเส้นทาง | ช่องว่าง auto vs escalated = **ราคาของความไม่แน่ใจ** (~32 เท่า) |
| Request แยกเส้นทาง | stacked — สัดส่วนฟรี vs จ่าย |
| ต้นทุนสะสม + calls | คู่กันเสมอ (1 escalation = 1 call) |

## ข้อควรระวัง

- **`--mock` ไม่ได้ยิง API** latency จำลองจากค่าที่วัดจริง (WCB 26ms / GPT 751ms / debate 4557ms)
  ใช้จูน concurrency ได้ฟรีไม่จำกัดรอบ แล้วค่อยยืนยันด้วย `--live` รอบเดียว
- **n=127 ไม่ใช่ traffic โปรดักชัน** dashboard นี้ตอบ "orchestration ถูกไหม / p95 เท่าไร"
  ไม่ได้ตอบ "ทน 10k req/s ไหม" — อย่าเคลมเกินนั้น
- ตัวเลข **F1 ที่รายงานได้จริงมาจาก `router.py`** (leave-fold-out) ไม่ใช่จาก `loadtest.py`
  loadtest ใช้ threshold จากทั้งชุดเพราะมันวัด *ระบบ* ไม่ได้วัด *ความแม่น*
