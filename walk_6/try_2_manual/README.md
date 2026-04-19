# walk_6 try_2 — 신규 reward 강도 조정 (try_1 regression 후속)

날짜: 2026-04-18
출발 ckpt: walk_5 try_13 final (Apr18_09-39-52_walk_5, model_7950.pt) — try_1 정책 폐기

## try_1 결과 (실패)
| 지표 | walk_5 try_13 final | walk_6 try_1 final | 변화 |
|---|---|---|---|
| `mean_reward` | 170.98 | **111.38** | **-59.6 ❌** |
| `swing_height` (raw) | 0.412 | 0.412 | 동일 (개선 없음) |
| `diagonal_gait` | 0.85 | 0.50 | ↓ |
| `diagonal_propulsion` (raw) | — | 0.123 | 매우 낮음 (sigma 0.12 너무 좁음) |
| `straight_line_deviation` (raw) | — | 0.794 | drift 0.27m, 페널티 -1.19 |
| `left_right_symmetry` (raw) | 0.85 | 0.62 | ↓ |
| `tracking_lin_vel` | 1.27 | 1.08 | ↓ |
| `ang_vel_xy` | -0.05 | -0.33 | 자세 흔들림 ↑ |
| `action_noise_std` | 0.12 | **0.40** | 3.3× 폭증 (정책 발산) |

## try_1 실패 원인
1. **`straight_line_deviation` scale -1.5 너무 강함** — 누적 lateral drift는 episode 후반에 폭증.
   페널티가 시간에 따라 비선형 증가해 정책이 panic 회피 동작 (yaw drift, 자세 흔들림).
2. **`diagonal_propulsion` sigma 0.12 너무 좁음** — pair ratio 0.5에서 살짝 벗어나도 보상 거의 0.
3. **mix mode w_min=0.6 효과 미미** — mean이 worst foot에 끌려가 raw 0.412 그대로.
4. **entropy_coef 0.0050** + 신규 reward conflict → action_noise_std 0.40 폭증, 정책 발산.

## try_2 변경
| 파라미터 | try_1 | try_2 | 의도 |
|---|---|---|---|
| `swing_height_min_weight` | 0.60 | **0.40** | mean에 더 비중 |
| `straight_dev_sigma` | 0.30 | **0.60** | drift 더 허용 |
| `pair_force_sigma` | 0.12 | **0.20** | ratio 0.3~0.7도 부분 보상 |
| scale `straight_line_deviation` | -1.50 | **-0.50** | 페널티 1/3로 (혼란 방지) |
| scale `diagonal_propulsion` | 1.50 | **2.50** | sigma 확대 + scale↑로 균형 강화 |
| `entropy_coef` | 0.0050 | **0.0035** | 정책 안정화 (try_1 발산 방지) |

## 변경 안 함 (try_13 잘 작동, try_1에서도 손상 안됨)
- swing_height target 0.040, sigma 0.030, scale 6.0
- diagonal_gait scale 2.0, all_feet_stepping scale 2.6
- left_right_stance_symmetry scale 1.2
- control penalties (torques/dof_vel/dof_acc/action_rate)
- foot_stance_max_time 0.30, stuck_foot scale -5.0
- max_iterations 2000

## Resume
- `load_run`: `logs/walk_5/Apr18_09-39-52_walk_5`
- `checkpoint`: 7950 (walk_5 try_13 final)
- try_1 정책(`logs/walk_6/Apr18_11-03-57_walk_6`)은 폐기

## try_2 모니터 포인트
- [ ] `action_noise_std` 0.15 이하 유지 (정책 안정)
- [ ] `mean_reward` ≥ 160 (walk_5 try_13 수준 회복)
- [ ] `rew_swing_height` raw > 0.45 (mix mode w_min=0.4 효과)
- [ ] `rew_diagonal_propulsion` raw > 0.5 (sigma 확대로)
- [ ] `rew_straight_line_deviation` |raw| < 0.4 (drift < 0.50m)
- [ ] `rew_diagonal_gait` raw > 0.40 유지
- [ ] `rew_left_right_stance_symmetry` raw > 0.75 유지

## 실패 시 (try_3 후보)
- 여전히 swing_height plateau: w_min=0.2까지 더 낮춤
- straight_line 여전히 큼: sigma 0.80으로 더 완화
- propulsion 여전히 낮음: ratio 정의를 force 대신 *swing time* 비율로 변경 검토
- 또는 walk_5 try_13으로 후퇴 + 신규 reward 도입을 walk_7로 미룸
