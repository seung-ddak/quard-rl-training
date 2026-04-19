# walk_6 try_6 — anti-shuffle 강제 + 덜 굳은 ckpt에서 재학습

날짜: 2026-04-18 (준비 완료, 학습은 내일 사용자가 실행)

## try_5 결과 정리 (학습 중단 시점, iter 14288)
- `mean_reward = 205.27` (try_4 final 197 초과 회복)
- `straight_line_deviation` raw 0.165 (drift 0.24m, 회복)
- `diagonal_propulsion` raw 0.87 유지
- **`swing_air_height` raw 0.0017 / `swing_height` raw 0.326** ← 발 들기 여전히 미미
- 디버그 결과: 정지/airborne 발 z 차이가 **0.03 mm** 수준 — 정책이 사실상 발을 안 들고 shuffle로 전진

## try_6 핵심 진단
walk_4 → walk_5 → walk_6 try_4/5 모두 거치며 정책이 **"발 안 들고 살짝 미끄러져 전진"** 국부 최적해에 굳어버림.
- foot_z 좌표계 버그(음수)는 try_5에서 수정 — `_max_swing_height` 정상 작동
- 그러나 정책 자체가 lift를 0.03mm로 수렴 → swing 보상은 작동해도 정책이 거기서 이동 안 함

## try_6 변경

### 1. 신규 reward `_reward_no_shuffle` (scale -4.0)
```python
max_lift = max(foot_z - min(foot_z) across feet)
no_lift_weight = 1 - clamp(max_lift / 0.010, 0, 1)
forward_speed = clamp(base_lin_vel_x, min=0)
return move_cmd * no_lift_weight * forward_speed
```
- 4발 중 가장 높이 들린 발이 1cm 미만이면 forward speed에 비례 페널티
- 정책 선택: (a) 발 들기 (b) 멈추기. shuffle은 더 이상 보상 못 받음
- smooth ramp (1cm 임계 부근에서 점진적)

### 2. `feet_slip` / `rear_feet_slip` 페널티 대폭 강화
- `feet_slip` -5.0 → **-10.0** (try_2 -2.0 대비 5배)
- `rear_feet_slip` -3.5 → **-7.0**
- shuffle 시 slip 페널티가 너무 커서 더 이상 통하지 않음

### 3. `swing_air_height` 강화
- scale 4.0 → **8.0** (lift 신호 2배)

### 4. **출발 ckpt 변경**: walk_5 try_13 final
- try_4/5 final은 shuffle에 깊게 굳음 → 회복 어려움
- walk_5 try_13 final은 lift는 안 하지만 trot 위상은 살아있음, 더 가소성 높음
- `load_run = logs/walk_5/Apr18_09-39-52_walk_5`, `checkpoint = 7950`

### 5. 변경 안 함 (기능 보존)
- 좌표계 버그 수정 (try_5에서 적용된 `_update_gait_tracking` 그대로)
- `straight_line_deviation` -0.50, sigma 0.60
- `diagonal_propulsion` 2.50, sigma 0.20
- `diagonal_gait` 2.0, `all_feet_stepping` 2.6
- `left_right_stance_symmetry` 1.20, EMA mode
- `tracking_lin_vel` 1.20, `tracking_ang_vel` 1.00, `orientation` -2.40
- control penalty (try_3 65% 감)
- `entropy_coef` 0.0035, `max_iterations` 2000
- `swing_height` (legacy) scale 6.0
- `stuck_foot_penalty` -7.0, `foot_stance_max_time` 0.18

## 실행 (내일 사용자 직접)
```bash
cd /mnt/gstore/home/xiangyue/RL/legged_gym
LD_LIBRARY_PATH=/home/xiangyue/.conda/envs/isaacgym/lib \
nohup /home/xiangyue/.conda/envs/isaacgym/bin/python \
legged_gym/scripts/train.py --task quard_walk_6 --headless \
> /mnt/gstore/home/xiangyue/RL/training_record/walk_6/try_6_manual/train_log.txt 2>&1 &
```

## try_6 모니터 포인트
- [ ] `rew_no_shuffle` 초기에 큰 음수 (-50 이상) → 50~200 iter 내 절댓값 감소 (lift 학습 신호)
- [ ] `rew_swing_air_height` raw 0.0017 → **> 0.5** (실질 lift 발생)
- [ ] `rew_swing_height` (legacy) raw 0.325 → **> 0.5** (effective > 1cm)
- [ ] `mean_reward` ≥ 170 (walk_5 try_13 수준 회복)
- [ ] `tracking_lin_vel` > 1.0 유지 (no_shuffle 페널티가 정책을 멈춤으로 내몰지 않는지)
- [ ] `straight_line_deviation` raw < 0.20
- [ ] `diagonal_gait` raw > 0.45 유지
- [ ] play.py 육안: 4발이 실제로 들리는 walking motion

## 실패 시 (try_7 후보)
- `tracking_lin_vel`이 0.5 이하로 떨어짐 (정책이 멈춤): `no_shuffle` scale -4.0 → -2.0 완화
- `swing_air_height` 여전 미미: scale 8.0 → 12.0, swing target 0.030 → 0.025 하향
- 정책 발산 (action_noise_std > 0.20): entropy 0.0035 → 0.0025
- 직진성 손상: `straight_line_deviation` scale -0.50 → -0.80

## 코드 변경 위치 (디버깅용)
- `legged_robot.py`:
  - `_update_gait_tracking` (line ~189): foot_heights = clamp(z - min(z), 0) 추가
  - `_reward_swing_air_height`: ground_ref 기반
  - `_reward_no_shuffle`: 신규 (try_6)
- `quard_config.py` `QuardWalk6Cfg`:
  - `shuffle_lift_thr = 0.010` (신규)
  - `swing_air_height = 8.00`, `no_shuffle = -4.00`, `feet_slip = -10.00`, `rear_feet_slip = -7.00`
  - `load_run = logs/walk_5/Apr18_09-39-52_walk_5`, `checkpoint = 7950`
