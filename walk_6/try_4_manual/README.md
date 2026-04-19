# walk_6 try_4 — swing_height stuck-floor 돌파

날짜: 2026-04-18
출발 ckpt: walk_6 try_3 final (`Apr18_14-26-16_walk_6`, model_11950.pt)

## try_3 결과 — 모든 면 정점, swing만 floor에 갇힘
| 지표 | walk_5 try_13 | walk_6 try_2 | **walk_6 try_3** |
|---|---|---|---|
| `mean_reward` | 170.98 | 203.49 | **223.87** |
| `straight_line_deviation` (raw) | — | 0.039 | **0.045** (drift 0.16m) |
| `diagonal_propulsion` (raw) | — | 0.68 | **0.81** |
| `diagonal_gait` (raw) | 0.42 | 0.49 | **0.66** |
| `left_right_symmetry` (raw) | 0.85 | 0.93 | **0.93** |
| `swing_height` (raw) | 0.412 | 0.412 | **0.325** ← 모두 floor 값 |

## swing_height plateau 진단 (사용자 질문 대응)
표시값이 정확히 같은 값으로 freeze된 이유:
- min_mode 로직: `stuck = (foot_stance_time > swing_peak_stance_reset)` 일 때 effective→0
- 모든 발이 stance>0.30s에 머물면 effective=0 (4발 동시) → per_foot이 상수
- `per_foot = exp(-target²/2σ²)` 로 모든 envs/feet/steps에서 동일
- 즉 표시되는 raw는 **"발을 거의 안 들 때 받는 수학적 floor 보상"**
- try_2: exp(-0.040²/0.0018) = 0.411
- try_3: exp(-0.030²/0.0008) = 0.325
- 정책이 "feet stuck + xy slip 으로 전진" 국부 최적해에 수렴, swing_height만으론 못 빠져나옴

## try_4 변경

### 1. 신규 reward `_reward_swing_cycle_peak` (event-based, scale 4.0)
```python
touchdown = contact & (~prev_contact)     # lift→land 순간
peak_norm = clamp(_last_swing_peak / target, 0, 1.5)
return move_cmd * sum(peak_norm * touchdown, dim=feet)
```
- 보상이 touchdown 순간에만 fire → stuck 발은 영원히 0 reward
- floor 회피 가능: 발을 안 드는 정책엔 신호 없음 → 들 incentive
- peak 비례라 cycle 당 더 높이 들수록 더 큰 보상 (최대 1.5×)

### 2. `feet_slip` / `rear_feet_slip` 페널티 강화
- `feet_slip` -2.0 → **-5.0** (xy slip = 페널티 2.5×)
- `rear_feet_slip` -1.35 → **-3.50**
- 이유: 정책이 발을 끌면서 xy로 전진하는 shuffle을 직접 차단

### 3. stuck 임계 더 엄격
- `swing_peak_stance_reset` 0.30 → **0.20s** (effective→0 트리거 더 빨리)
- `foot_stance_max_time` 0.25 → **0.18s** (stuck 페널티 더 빨리)
- `stuck_foot_penalty` scale -5.0 → **-7.0**

### 4. legacy `swing_height` 보조 격하
- scale 8.0 → **3.0** (floor 보상 영향력 축소; 주력은 swing_cycle_peak)

### 5. 변경 안 함 (try_3 잘 작동, 보존)
- `straight_line_deviation` -0.50, sigma 0.60
- `diagonal_propulsion` 2.50, sigma 0.20
- `diagonal_gait` 2.0, `all_feet_stepping` 2.6, `left_right_stance_symmetry` 1.20
- `tracking_lin_vel` 1.20, `tracking_ang_vel` 1.00, `orientation` -2.40
- control penalty (try_3 65% 감 유지): torques -0.00007, dof_vel -0.0005, etc.
- `entropy_coef` 0.0035, `max_iterations` 2000

## Resume
- `load_run`: `logs/walk_6/Apr18_14-26-16_walk_6`
- `checkpoint`: 11950 (walk_6 try_3 final)

## 모니터 포인트
- [ ] `rew_swing_cycle_peak` raw > 0.5 (cycle당 평균 절반 이상 도달)
- [ ] `rew_swing_height` (legacy) raw > 0.40 (floor 0.325 초과)
- [ ] `mean_reward` > 200 유지
- [ ] `feet_slip` 절대값 ≤ 0.05 (slip 줄어들어야 진짜 발 들기)
- [ ] `stuck_foot_penalty` 0 근처 (느린 stance 0.18s 이내)
- [ ] `straight_line_deviation` raw < 0.10 유지
- [ ] `action_noise_std` < 0.15

## 실패 시 (try_5 후보)
- swing_cycle_peak 여전히 낮음: scale 4.0 → 6.0
- 정책 발산 (action_noise_std > 0.20): stuck 임계 완화 (0.18 → 0.22)
- 직진성 손상: feet_slip 약간 완화 (-5.0 → -3.5)
