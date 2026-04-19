# walk_6 try_3 — swing_height plateau 전용 돌파

날짜: 2026-04-18
출발 ckpt: walk_6 try_2 final (`Apr18_13-08-27_walk_6`, model_9950.pt)

## try_2 결과 (성공한 면)
| 지표 | walk_5 try_13 | walk_6 try_2 final | 평가 |
|---|---|---|---|
| `mean_reward` | 170.98 | **203.49** | +32.5 ✅ |
| `straight_line_deviation` (raw) | — | 0.039 (drift 0.13m) | ⭐ 거의 완벽 직진 |
| `diagonal_propulsion` (raw) | — | 0.675 | ✅ 양 쌍 균형 |
| `left_right_stance_symmetry` (raw) | 0.85 | **0.93** | ✅ try_13 초과 |
| `diagonal_gait` (raw) | 0.42 | **0.49** | ✅ try_13 초과 |
| `tracking_lin_vel` | 1.27 | 1.17 | OK |
| `orientation` | -0.04 | -0.003 | ✅ 매우 안정 |
| `action_noise_std` | 0.12 | 0.10 | ✅ 안정 |
| **`swing_height` (raw)** | **0.412** | **0.412** | ❌ **유일한 미해결 plateau** |

## 진단 (swing_height plateau 끈질긴 이유)
- mix mode w_min=0.4도 효과 없음 → min과 mean이 거의 같은 값
- 즉 **모든 발이 비슷하게 effective ≈ 0~0.025m에서 약간만 듦**
- 발을 더 들면 즉시 control penalty(`torques`, `dof_vel`, `dof_acc`, `action_rate`) 증가
- swing_height scale 6.0 vs 합산 control penalty가 동등해서 정책이 더 들 incentive 없음
- try_13에서 control penalty 30% 완화로 0.279→0.412 돌파했지만 그 이상은 안 됨

## try_3 변경 (swing_height만 집중)

### swing_height 자체 강화
| 파라미터 | try_2 | try_3 | 의도 |
|---|---|---|---|
| `swing_height_target` | 0.040 | **0.030** | 정책이 도달 가능한 높이로 |
| `swing_height_sigma` | 0.030 | **0.020** | target 낮춘 만큼 좁혀 gradient 유지 |
| `swing_height_min_weight` | 0.40 | **0.30** | mean에 더 비중 (mix 효과 강화) |
| `foot_stance_max_time` | 0.30 | **0.25** | 더 짧은 stance, lift cycle 증가 |
| scale `swing_height` | 6.00 | **8.00** | 더 강한 pull |

### Control penalty 추가 50% 완화 (try_13 대비 65% 감)
발 들기 비용 자체를 거의 제거 → swing_height 보상이 압도하도록.
| 파라미터 | try_2 | try_3 |
|---|---|---|
| `torques` | -0.00014 | **-0.00007** |
| `dof_vel` | -0.0010 | **-0.0005** |
| `dof_acc` | -2.8e-7 | **-1.4e-7** |
| `action_rate` | -0.011 | **-0.006** |

### 변경 안 함 (try_2에서 잘 작동, 보존)
- `straight_line_deviation` scale -0.50, sigma 0.60 (drift 0.13m 유지)
- `diagonal_propulsion` scale 2.50, sigma 0.20 (양 쌍 균형 유지)
- `left_right_stance_symmetry` scale 1.20, EMA mode (raw 0.93 유지)
- `diagonal_gait` scale 2.00, `all_feet_stepping` scale 2.60
- `tripod_penalty`, `stuck_foot_penalty` 그대로
- `entropy_coef` 0.0035, `max_iterations` 2000

## Resume
- `load_run`: `logs/walk_6/Apr18_13-08-27_walk_6`
- `checkpoint`: 9950 (walk_6 try_2 final, 모든 좋은 면 보존된 정책)

## 모니터 포인트
- [ ] `rew_swing_height` raw **≥ 0.55** (target 0.030 기준, 발이 실제로 ~2cm 들어야)
- [ ] `mean_reward` ≥ 200 유지 (try_2 수준)
- [ ] `straight_line_deviation` raw < 0.10 유지
- [ ] `diagonal_propulsion` raw > 0.6 유지
- [ ] `left_right_symmetry` raw > 0.85 유지
- [ ] `action_noise_std` < 0.15 (control penalty 완화로 발산하지 않는지)

## 실패 시 (try_4 후보)
- swing_height 여전히 plateau:
  - target 0.030→0.025로 더 낮춤
  - control penalty 추가 50% (총 80%+ 감)
  - 또는 swing_height reward 함수 자체 재작성: 누적 페널티가 아니라 cycle당 max-clearance 직접 보상
- 정책 발산 (action_noise_std > 0.20):
  - control penalty 완화 폭 줄임 (50% → 30%)
  - entropy_coef 0.0035 → 0.0030
