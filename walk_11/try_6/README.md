# walk_11 try_6 — sharpening on try_5 peak (entropy ↓ + diagonal_gait ↑)

## 사용자 지시 (2026-04-28)

> "try 6 play now" — try_5 peak 후 regression 진정 + trot sharpening

## 배경

try_5 결과: peak iter 10549 (rew 28.91, ep_len 720, diag_gait 1.13) 후 regression (rew 24.59, ep_len 647, noise std 0.63). soft thigh penalty (-2) + clean baseline 조합은 성공했으나 entropy_coef 0.008 이 후반엔 과한 탐색 → noise std 폭주 + trot 어색함 잔존.

## 변경 (`quard_config.py:559-580`)

| 항목 | try_5 | try_6 | 의도 |
|---|---|---|---|
| baseline | walk_10 try_17 model_9950 (Apr26_14-31-05) | **walk_11 try_5 model_10550 (peak)** (Apr27_14-09-56) | regression 전 정점에서 재출발 |
| `entropy_coef` | 0.008 | **0.005** | noise std 0.63 → ~0.4 진정, policy sharpening |
| `diagonal_gait` scale | 2.0 | **2.5** (+25%) | 어색한 trot 회복 강화 |
| `max_iterations` | 2000 | **1000** | sharpen 만, 과학습 방지 |
| 그 외 | (그대로) | (그대로) | terrain rough_slope, DR, collision -2, thigh hard term |

## 학습 설정

- baseline ckpt: `logs/walk_11/Apr27_14-09-56_walk_11/model_10550.pt` (try_5 peak)
- 학습 타겟 iter: 10550 → 11550 (1000 iter)
- 새 log dir: `logs/walk_11/Apr28_05-25-53_walk_11/`

## 결과 (2026-04-28 학습 완료)

| 항목 | 값 | 평가 |
|---|---|---|
| 총 iteration | **11549 / 11550** (정상 종료) | ✅ |
| 학습 시간 | **5080초 (84.7분 ≈ 1.41시간)** | — |
| 총 timesteps | 98,304,000 | — |
| Mean reward (peak iter 10800) | **32.59** | ✅ try_5 peak (28.91) 대비 **+13%** |
| Mean reward (final iter 11549) | **20.21** | ⚠️ peak 대비 -38% regression |
| ep_len (peak iter 10800) | **610.78** | ✅ |
| ep_len (final) | **414.17** | ⚠️ peak 대비 -32% |
| **noise std (final)** | **0.39** | ✅ try_5 (0.63) 대비 -38%, sharpening 성공 |
| diagonal_gait (peak) | **1.25** | ✅ |
| diagonal_gait (final) | 0.92 | ✅ 1.0 근접 유지 |
| diagonal_propulsion (final) | 0.61 | ⚠️ try_5 final 0.82 보다 낮음 |
| feet_slip (final) | -0.63 | — |
| **collision (thigh)** | **-0.0017** | ✅ try_5 (-0.0013) 수준, grazing 거의 없음 |
| terrain_level | 0.0000 | ⚠️ 진급 X (walk_10/11 구조적 한계) |
| 최종 ckpt | `model_11550.pt` | regression 박힌 정책 |
| **Peak ckpt** | `model_10800.pt` | ✅ try_6 의 진짜 best 후보 |

## 학습 진행 (요약)

- iter 10550 (start): rew -0.13, ep_len 18 (예상된 transition shock — entropy ↓ + diagonal_gait ↑ 첫 노출)
- iter 10600: rew 19.56, ep_len 469, diag 0.96 (빠른 회복)
- iter 10700: rew 28.36, ep_len 594, diag 1.29
- **iter 10800 peak**: rew **32.59**, ep_len 611, diag 1.25 ← clean walking 정점, try_5 peak 초과
- iter 10900: rew 28.12 dip 시작
- iter 11000~11200: rew 22 → 17 → 17 (bottom dip), ep_len 350~450
- iter 11300~11549: rew 21~22 oscillation, ep_len 410~440 (천천히 회복했으나 peak 미달)

## try_5 / try_6 비교

| 항목 | try_5 (final) | try_6 (final) | 변화 |
|---|---|---|---|
| Mean reward | 24.59 | 20.21 | ⚠️ -18% |
| ep_len | 647 | 414 | ⚠️ -36% |
| noise std | 0.63 | 0.39 | ✅ -38% (sharpening) |
| diag_gait | 0.99 | 0.92 | ≈ 동등 |
| diag_propulsion | 0.82 | 0.61 | ⚠️ -26% |
| collision | -0.0013 | -0.0017 | ≈ 동등 |
| Peak (best) reward | 28.91 (iter 10549) | **32.59 (iter 10800)** | ✅ +13% |

→ **final 만 보면 try_5 가 더 좋음**, **peak 만 보면 try_6 가 더 좋음**. 같은 regression 패턴이 더 빨리 (250 iter 만에) 일어났음.

## 분석

### 좋은 점

- **noise std 0.63 → 0.39 진정 성공**: entropy_coef 0.005 변경이 의도대로 작동, 탐색 줄이고 활용 강화
- **새 peak 달성 (rew 32.59)**: try_5 peak 28.91 초과, diagonal_gait 2.5 강화 효과 확인
- **thigh grazing 유지**: collision -0.0017, try_5 수준
- 1000 iter 로 짧게 끊은 게 맞는 선택 (regression 박히기 전에 종료)

### 한계 / 의문

- **iter 10800 peak 후 또 regression**: try_5 (10549 peak) 와 패턴 동일, 더 빨리 발생 (250 iter vs 400 iter). 원인 가설:
  - 짧은 학습 기간에 비해 diagonal_gait 2.5 가 너무 강함 → 다른 reward (tracking_lin_vel, base_height) 와 균형 깨짐
  - PPO clipping 안에서 entropy 너무 낮아 local optimum 탈출 못함
- **diag_propulsion 0.61 ↓**: trot 이 깔끔하지만 추진력 약함. diagonal_gait reward 만 올리고 propulsion 관련은 그대로여서 균형 흐트러진 듯
- **terrain_level = 0**: walk_10/11 공통 한계, walk_12 영역

## 다음 단계 후보

- **play.py 육안 비교** (사용자 진행 예정): `model_10800` (try_6 peak) vs `model_10550` (try_5 peak) vs `model_11550` (try_6 final)
- 만약 model_10800 이 육안으로 best → real_baseline 갱신 후 walk_12 진행
- 만약 trot 어색함 잔존 → diagonal_gait 2.5 → 2.2 로 살짝 낮추거나, propulsion reward 동시 강화 (try_7)
- regression 본질적으로 막으려면: cosine LR schedule 도입 또는 max_iter 500 으로 더 짧게

## ckpt 위치

- Peak (best 후보): `model_10800.pt` (try_6/ 와 logs/walk_11/Apr28_05-25-53_walk_11/ 둘 다)
- Final: `model_11550.pt` (try_6/ 와 동일 log dir)
- Logs: `logs/walk_11/Apr28_05-25-53_walk_11/`
