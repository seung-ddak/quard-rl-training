# walk_6 try_1 — 품질 재설계 (Quality Redesign)

날짜: 2026-04-18
출발 ckpt: walk_5 try_13 final (`Apr18_09-39-52_walk_5`, model_7950.pt)

## 출발점 (walk_5 try_13 final, iter 7949)
- mean_reward 170.98 ✅
- swing_height raw **0.412 plateau** (min mode cap, worst foot ≈ 0)
- diagonal_gait raw 0.42, all_feet_stepping raw 0.36
- left_right_symmetry raw 0.85, stuck_foot 0
- tracking_lin_vel 1.27, orientation -0.04
- 직진 deviation 페널티 없음 → 평지 외엔 휘어 나갈 위험

## walk_6 신규 reward (코드 변경 포함)

### 1. `_reward_swing_height` MIX 모드
**문제:** min over feet은 worst foot 하나가 0이면 raw가 0.41로 cap됨.
**해결:** `swing_height_min_weight = 0.6` 도입.
```
mixed = w_min·min(per_foot) + (1-w_min)·mean(per_foot)
```
- worst foot이 0이라도 다른 3발이 잘 들면 부분 보상 → 정책이 worst foot을 점차 들도록 유도
- w_min=1.0이면 legacy min mode (try_13와 동일)

### 2. `_reward_straight_line_deviation` 신규
spawn 위치 + spawn forward를 reset 시 저장. 매 step마다 spawn 기준 lateral drift 측정:
```
delta = root_xy - spawn_xy
lateral = delta · perp(spawn_forward)
return lateral² / sigma²    (positive scale = -1.5 → 음수 보상)
```
- 평지 학습 정책이 거친 지형에서도 직진 유지하도록 유도
- 순간 sway는 약하게, 누적 drift는 강하게 페널티

### 3. `_reward_diagonal_propulsion` 신규
양 대각쌍의 수직 지면반력(z-force) EMA가 0.5에 가깝도록:
```
pair_a_force = LF_z + RR_z
pair_b_force = LR_z + RF_z
ratio_ema = EMA(pair_a / (pair_a + pair_b), alpha=0.02)
return exp(-(ratio_ema - 0.5)² / 2σ²)
```
- 한 쌍만 차고 반대 쌍이 scuff하는 try_12/13의 잔존 결함 직접 페널티
- σ=0.12 → ratio가 0.38~0.62 안에서 약 0.6 보상

## 코드 변경 요약 (`legged_gym/envs/base/legged_robot.py`)
- `_init_buffers`: `_spawn_xy`, `_spawn_forward`, `_pair_force_ema` 추가
- `reset_idx`: 위 3개 버퍼 리셋
- `_reward_swing_height`: `swing_height_min_weight` cfg-gate 추가
- `_reward_straight_line_deviation` 신규
- `_reward_diagonal_propulsion` 신규
(상세 스냅샷: `legged_robot_snapshot.py`)

## cfg 변경 요약 (`quard_config.py` walk_6)
| 항목 | 값 | 비고 |
|---|---|---|
| `swing_height_min_weight` | 0.60 | NEW: mix mode |
| `straight_dev_sigma` | 0.30 | NEW |
| `pair_force_sigma` | 0.12 | NEW |
| `pair_force_ema_alpha` | 0.02 | NEW |
| scale `straight_line_deviation` | -1.50 | NEW |
| scale `diagonal_propulsion` | 1.50 | NEW |
| 기타 scale/param | walk_5 try_13와 동일 | scaffolding 보존 |
| `domain_rand` | 모두 False | 품질 단계, DR은 walk_7로 미룸 |
| `lin_vel_x` | [0.28, 0.50] | 직진 명령 |

## Resume
- `load_run`: `logs/walk_5/Apr18_09-39-52_walk_5`
- `checkpoint`: 7950
- `max_iterations`: 2000

## 실행
```bash
LD_LIBRARY_PATH=/home/xiangyue/.conda/envs/isaacgym/lib \
/home/xiangyue/.conda/envs/isaacgym/bin/python \
legged_gym/scripts/train.py --task quard_walk_6 --headless
```

## try_1 모니터 포인트
- [ ] `rew_swing_height` raw 0.41 → **≥ 0.55** (mix mode 효과)
- [ ] `rew_diagonal_propulsion` raw > 0.7 (양 쌍 균형)
- [ ] `rew_straight_line_deviation` |raw| < 1.0 (lateral drift < 0.30m)
- [ ] `rew_diagonal_gait` raw 유지 (>0.40)
- [ ] `rew_left_right_stance_symmetry` raw 유지 (>0.80)
- [ ] `rew_tracking_lin_vel` > 1.1
- [ ] `rew_stuck_foot_penalty` 0 근처
- [ ] play.py 육안: 직선 유지 + 양 대각쌍 모두 들고 차는 동작

## 실패 시
- swing_height raw < 0.45: w_min 더 낮춤 (0.4) + sigma 0.035로 확대
- diagonal_propulsion 폭주: sigma 0.15로 확대
- straight_line_deviation 폭주: sigma 0.40으로 확대 (drift 더 허용)
- 모두 실패: walk_6 try_2에서 swing_height target 0.040→0.035 하향
