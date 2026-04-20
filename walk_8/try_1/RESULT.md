# walk_8 try_1 — Front-Rear Symmetry 보강 결과

**2026-04-20 학습** · model_12250.pt (Apr20 본 학습)

## 결과 요약
**합격 기준 전부 통과**. walk_7을 넘어선 새 best.

| 지표 | walk_7 try_2 | walk_8 try_1 | 변화 | 합격선 |
|---|---:|---:|---:|---:|
| Mean reward | 164.65 | **170.12** | +5.47 | ≥155 ✅ |
| front_rear_stance_symmetry | 0.2082 | **0.5216** | ×2.51 | ≥0.50 ✅ |
| left_right_stance_symmetry | 1.0067 | 0.9757 | -0.031 | ≥0.90 ✅ |
| swing_height | 1.9543 | 1.9543 | 동일 | 유지 ✅ |
| episode length | 1002 | 1002 | max | 조기종료 없음 |
| tracking_lin_vel | 0.8858 | 0.8097 | -0.076 | 허용 범위 |
| tracking_ang_vel | 0.9183 | 0.9378 | +0.020 | - |
| diagonal_gait | 0.6042 | 0.7042 | +0.100 | - |

## 개입 내용
- `front_rear_stance_symmetry` scale **0.25 → 0.60** (2.4x)
- 그 외 reward / DR / env 모두 walk_7 try_2와 동일
- walk_7 try_2 model_11250 에서 resume, 1000 iter 학습

## 학습 곡선 (Mean reward)
| iter | reward |
|---:|---:|
| 1 | 13.12 (reset) |
| 50 | 5.40 (reset 직후) |
| 200 | 140.12 |
| 500 | 164.24 |
| 800 | 168.93 |
| 950 | 171.05 (peak) |
| 995 | 170.12 (final) |

Plateau 없음, 안정적 수렴 (최종 5 iter σ<0.5).

## 해석
1. front_rear scale 상향이 정확히 효과 발휘 — 좌우 대칭(1.00)에 못 미치던 앞뒤 대칭이 0.52로 대폭 개선
2. 놀랍게도 reward 전반 상승 (+5.47) — 앞뒤 대칭이 다른 보상과 synergy
3. tracking_lin_vel 미세 저하 (-0.08)는 gait 재조정으로 인한 trade-off, 허용 범위

## 다음 단계 후보
- **walk_8 try_2**: front_rear scale 0.60 → 0.90 추가 상향 (0.70+ 달성 시도)
- **walk_9**: lift 돌파 (`swing_cycle_peak` 재활성 또는 stance time curriculum)
- **walk_9-alt**: DR 강도 확대 (friction [0.3, 1.7], push 0.5 m/s)

현재 best: **walk_8 try_1 model_12250**
