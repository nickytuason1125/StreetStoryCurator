# Taste agreement report
_2026-09-05 16:37 — 671 rated photos with score snapshots (124 legacy ratings without snapshots excluded)_

## 1. Machine grader vs your stars
- Bucket agreement: **637/671 = 94.9%**

| your stars → | machine: Strong | machine: Mid | machine: Weak |
|---|---|---|---|
| Strong | 55 | 0 | 0 |
| Mid | 0 | 304 | 34 |
| Weak | 0 | 0 | 278 |

## 2. Taste head discrimination (AUC)
- **AUC = 0.848** over 55x278 = 15290 high/low pairs (strong)
- (0.50 = coin flip, 1.00 = your head ranks every photo you rated highly above every photo you rated low)

## 3. Monotonicity — mean scores per star level

| stars | n | machine score | taste score |
|---|---|---|---|
| 2★ | 278 | 0.328 | 0.449 |
| 3★ | 338 | 0.478 | 0.502 |
| 4★ | 53 | 0.658 | 0.579 |
| 5★ | 2 | 0.650 | 0.562 |

## Reading it
- Machine agreement is the grader's baseline trust level.
- Taste AUC is the PersonalHead's — the number the 0.20→0.70
  confidence-adaptive blend earns its authority with.
- Monotonicity should climb with stars. If it doesn't, the
  head is learning noise, not you.
