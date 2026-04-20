# walk_9 try_1 — Domain Randomization 확대 성공

**2026-04-20 학습** · model_13250.pt · **새 best**

## 개입
- friction_range [0.5, 1.5] → **[0.3, 1.7]**
- max_push_vel_xy 0.3 → **0.5**
- push_interval_s 8.0 → **6.0** (더 자주)
- front_rear scale 0.60 유지 (walk_8 try_1 설정)
- walk_8 try_1 model_12250 (mean 170.12, stable) resume

## 결과 — 합격 기준 전부 통과 + 신기록

| 지표 | walk_8 try_1 | **walk_9 try_1** | 변화 | 합격선 |
|---|---:|---:|---:|:-:|
| **Mean reward** | 170.12 | **172.02** | **+1.90** | ≥155 ✅ |
| front_rear_sym | 0.5216 | 0.4996 | -0.02 | ≥0.40 ✅ |
| left_right_sym | 0.9757 | **0.9804** | +0.005 | ≥0.85 ✅ |
| swing_height | 1.9543 | 1.9541 | 동일 | ✅ |
| episode_length | 1002 | 1002 | max | ≥900 ✅ |
| **diagonal_propulsion** | 1.4595 | **1.9953** | **+0.54** | - |
| tracking_lin_vel | 0.8097 | 0.8774 | +0.07 | - |
| tracking_ang_vel | 0.9378 | 0.8610 | -0.08 | - |
| Value loss | 0.0114 | 0.1033 | 소폭 ↑ | - |

## 수렴 곡선
| iter | reward |
|---:|---:|
| 1 | 5.71 (reset) |
| 200 | 152.29 |
| 500 | 162.42 |
| 800 | 164.76 |
| 900 | 159.23 (dip, DR 적응) |
| 950 | 169.42 |
| 988 | 170.62 |
| 994 | **172.02** (최종) |

**최종 7 iter σ ≈ 0.6**, 상승세 유지 후 수렴. 안정.

## 예상 밖 수확
1. **중간 0.60 → 최종 0.98 회복**: iter 902 체크 시 left_right_sym 0.60이었으나 최종 0.98로 복구 — DR 적응 과정의 transient
2. **diagonal_propulsion +0.54**: 추진력 대폭 상승 (보행 효율성 향상)
3. **tracking_lin_vel +0.07**: 명령 속도 추종 개선
4. Mean reward +1.90: DR 강화가 오히려 regularization + 강건성으로 성능 향상

## 해석
walk_7 → walk_8 때 DR 추가가 성능 향상시킨 효과의 **연장선**.
friction 극단값 [0.3, 1.7]로 확대해도 정책이 적응하며 더 robust한 gait 학습.

## 현재 최종 best
**walk_9 try_1 model_13250** (mean 172.02, left_right 0.98, front_rear 0.50, DR 강화 대응).

## 다음 단계 후보
- walk_9 try_2: DR 더 확대 (friction [0.2, 1.8], push 0.7) — 한계 탐색
- walk_10: 지형 randomization (terrain) 또는 mass randomization
- walk_10: lift 재도전 (다른 접근) — 현재 swing_height 1.95 포화
