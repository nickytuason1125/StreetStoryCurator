# API Latency Benchmark

Base: `http://127.0.0.1:8000` · Generated: 2026-08-31 13:11:24

| Endpoint | n | p50 (ms) | p90 | p99 | max | bytes |
|---|---|---|---|---|---|---|
| health_warm | 12 | 2.7 | 2.8 | 2.9 | 2.9 | — |
| engine_health | 4 | 1.8 | 1.9 | 1.9 | 1.9 | — |
| catalog_slim | 10 | 99.9 | 106.6 | 106.6 | 106.6 | 35948552 |
| catalog_full | 2 | 2088.5 | 2088.5 | 2088.5 | 2088.5 | 53593173 |
| photo_detail_hot | 10 | 1.8 | 2.3 | 2.3 | 2.3 | — |
| thumb_first_touch | 20 | 3.2 | 4.2 | 4.2 | 4.2 | — |
| thumb_warm_avg | 5 | 2.5 | 2.6 | 2.6 | 2.6 | 4042 |
| star_roundtrip | 3 | 513.4 | 779.3 | 779.3 | 779.3 | — |
| places | 4 | 2.6 | 23.0 | 23.0 | 23.0 | — |
