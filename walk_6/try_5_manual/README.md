# walk_6 try_5 — ★ 장기 잠복 버그 수정 (foot height 좌표계) + air_height 신규

## ⚠️ 핵심 발견 (2026-04-18 디버그 결과)

`foot_positions[:,:,2]`가 음수 값 (-0.133)을 반환함.
- calf 링크 origin이 base 기준 -0.13m offset
- 정지 시 foot_z[i] ≈ -0.133 (모든 발 거의 동일)
- 들어 올린 발: foot_z ≈ -0.10 정도

**이 좌표계 버그가 walk_5/6 내내 swing_height plateau의 진짜 원인이었음:**
- `_max_swing_height = max(0, foot_z=음수) = 0` 영원히
- → `_last_swing_peak = 0` 영원히
- → 모든 swing 측정값이 0 → exp(-target²/2σ²)이라는 수학적 floor 보상만 발생
- → 정책이 발을 들어도 신호가 없어 학습 못함

즉 "정책이 floor에 갇힘"이 아니라 **reward signal 자체가 깨져 있었음**.
walk_5 try_13~walk_6 try_4의 swing_height raw 0.41 / 0.325는 정책 문제가 아니라 코드 버그.

## try_5 fix
1. **`_update_gait_tracking` 수정**: `foot_heights = clamp(foot_z - min(foot_z), 0)`로 ground-relative 측정.
   stuck 발 = lift 0, 들린 발 = 양수 lift. legacy swing_height도 자동으로 정상화.
2. **`_reward_swing_air_height` 추가**: 매 step, airborne 발의 lift 비례 보상.
   (스냅샷의 함수는 같은 ground_ref 방식 사용)

날짜: 2026-04-18
출발 ckpt: walk_6 try_4 final (`Apr18_16-05-41_walk_6`, model_13950.pt)

## try_4 결과 — cycle reward 버그
- `mean_reward = 197.15` (try_3 대비 -27, swing_height scale 격하 영향)
- `rew_swing_cycle_peak = 0.0000` ← 학습 내내 0 (silent fail)
- 다른 신호: `feet_slip` 절반 ↓ (slip 강화 효과), `stuck_foot` 0, `propulsion` 0.89, `straight` raw 0.018
- swing_height (legacy) 여전히 floor 0.325

`_reward_swing_cycle_peak` 진단:
- touchdown event 기반 (lift→land 순간에만 fire)
- `_last_swing_peak` 의존 — `_update_gait_tracking`이 touchdown 시 freeze하지만, 정책이 stuck floor에 갇힌 상태에선 envs마다 touchdown 빈도/타이밍이 매우 sparse하고 `_max_swing_height` 누적 상태가 불안정
- 결과: 전 학습 0.0000

## try_5 변경

### 핵심: `_reward_swing_air_height` 신규 (continuous, robust)
```python
airborne = ~contact
foot_z = self.foot_positions[:, :, 2]
height_norm = clamp(foot_z / target, 0, 1.5)
return move_cmd * sum(height_norm * airborne, dim=feet)
```
- 매 step, 매 발: airborne이면 height에 비례 reward, stance면 0
- stuck 발 → 영영 0 reward (touchdown event 의존 없음)
- shuffle 시 발이 ground에 붙어있어 보상 0 (foot_z near 0)
- floor 회피 가능: airborne이라는 명확한 활성화 조건

### scale & 파라미터
- `swing_air_height` scale **5.00** 신규
- `swing_air_target = 0.030`
- `swing_cycle_peak` scale **0.00** (try_4 함수 비활성, 코드 보존)
- 나머지는 try_4 그대로 (slip 강화, stuck 임계, control penalty 65% 감)

### 변경 안 함 (try_4 잘 작동)
- `feet_slip` -5.0, `rear_feet_slip` -3.50 (shuffle 차단 효과 확인)
- `stuck_foot_penalty` -7.0
- `swing_peak_stance_reset` 0.20, `foot_stance_max_time` 0.18
- `swing_height` (legacy) scale 3.0 (보조)
- `straight_line_deviation` -0.50, `diagonal_propulsion` 2.50
- control penalty (try_3 65% 감)

## Resume
- `load_run`: `logs/walk_6/Apr18_16-05-41_walk_6`
- `checkpoint`: 13950 (walk_6 try_4 final)
- max_iterations: 2000

## 모니터 포인트
- [ ] `rew_swing_air_height` raw > 0.5 (cycle reward와 달리 0 안 나와야 정상)
- [ ] `rew_swing_height` (legacy) raw > 0.40 (floor 0.325 초과)
- [ ] `mean_reward` > 200
- [ ] `feet_slip` 절대값 < 0.10 (shuffle 억제 유지)
- [ ] `stuck_foot_penalty` 0 근처
- [ ] `straight_line_deviation` raw < 0.10 유지
- [ ] `diagonal_propulsion` raw > 0.7 유지

## 실패 시 (try_6 후보)
- swing_air_height raw < 0.3: scale 5 → 8, swing_height (legacy) scale 3 → 0 (완전 격하)
- 정책 발산 (action_noise_std > 0.15): control penalty 일부 복구
- straight 손상: scale -0.5 → -0.7
