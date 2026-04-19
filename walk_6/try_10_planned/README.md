# walk_6 try_10 — anti-shuffle 최초 실행 (walk_5 try_13 7950 rollback + no_shuffle)

날짜: 2026-04-19

## 배경 — try_6 ~ try_9 전수 실패 분석
try_6, 7, 8, 9는 모두 "anti-shuffle 도입 전"의 walk_6 코드/config로 실행됨.
학습 로그에 `rew_no_shuffle` 자체가 등장하지 않음 → 실제로는 구버전 reward 셋.
→ 모두 **"발 안 들고 미끄러져 전진"** shuffle 국부 최적해로 수렴.

| try | iter 도달 | mean_reward | swing_air_height (raw) | 비고 |
|-----|---------|-------------|-------------------------|-----|
| 6   | 14333   | 191         | 0.0000                 | 완전 shuffle |
| 7   | 14251   | 188         | 0.0000                 | 완전 shuffle |
| 8   | 14015   | 34          | 0.0067                 | straight_line_deviation -8.18로 붕괴 |
| 9   | 14289   | 207         | 0.0069                 | 최고 점수지만 lift 1mm 미만 |

try_8 `[DBG swing_air_height]` 출력 확인:
```
foot_z[0]=[-0.1329, -0.1329, -0.1329, -0.1329]  # 4발 모두 같은 z
airborne[0]=[0.0, 1.0, 1.0, 0.0]                 # 2발은 airborne이라 판정되지만
height_norm[0]=[0.0, 0.0, 0.0, 0.0]              # 실제 lift 0
```
→ contact 판정상 airborne이지만 z 차이가 없는 "ground-grazing" shuffle이 완성됨.

## try_10 핵심 방침
1. **checkpoint rollback**: try_6~9 모두 shuffle-locked → `walk_5/Apr18_09-39-52_walk_5/model_7950.pt`로 복귀. walk_5 try_13 final은 lift는 안 하지만 trot 위상은 살아있음, 덜 굳음.
2. **신규 reward `_reward_no_shuffle` 활성화** (scale -4.0):
   - 4발 중 max lift < 1cm이면 forward speed에 비례 페널티
   - Smooth ramp (1cm 임계 부근 점진적)
3. **slip 페널티 강화** `feet_slip -10`, `rear_feet_slip -7` (shuffle 시 slip 직접 페널티)
4. **swing_air_height scale 4→8** (lift 신호 2배)

## 현재 config 상태 (변경 없음, 이미 준비됨)
`quard_config.py::QuardWalk6Cfg`:
- `shuffle_lift_thr = 0.010`
- `scales.swing_air_height = 8.00`
- `scales.no_shuffle = -4.00`
- `scales.feet_slip = -10.00`, `rear_feet_slip = -7.00`
- `scales.swing_height = 6.00` (legacy, 보조)
- `load_run = logs/walk_5/Apr18_09-39-52_walk_5`, `checkpoint = 7950`
- `entropy_coef = 0.0035`, `max_iterations = 2000`

`legged_robot.py`:
- `_reward_no_shuffle` 1458~1475줄 구현 완료 (ground_ref 기반)
- `_reward_swing_air_height` 1477~1492줄 (ground_ref 기반)
- `_update_gait_tracking` 189~210줄 (ground_ref clamp 수정)

## 모니터 포인트
- [ ] `rew_no_shuffle` 초기에 큰 음수 (-50 이상) → 50~200 iter 내 절댓값 감소
- [ ] `rew_swing_air_height` raw > 0.5 (실질 lift 발생)
- [ ] `rew_swing_height` (legacy) raw > 0.5 (effective > 1cm)
- [ ] `mean_reward` ≥ 170 (walk_5 try_13 수준)
- [ ] `tracking_lin_vel` > 1.0 (정책이 멈춤으로 내몰리지 않음)
- [ ] `straight_line_deviation` raw < 0.20
- [ ] `diagonal_gait` raw > 0.45 유지

## 실패 시 대응 (try_11)
- `tracking_lin_vel < 0.5`: `no_shuffle` -4.0 → -2.0 완화
- `swing_air_height` 여전 미미: scale 8→12, `shuffle_lift_thr` 0.010 → 0.008
- action_noise_std > 0.20 발산: `entropy_coef` 0.0035 → 0.0025
