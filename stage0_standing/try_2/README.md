# Stage 0 - Try 2: 안정적인 서기 자세 학습 (보상 구조 개선)

## Try 1 대비 변경사항
Try 1에서 뒷다리가 바닥에 닿지 않는 문제 → 보상 구조를 전면 재설계

### 새 보상 구조
**Positive Rewards:**
- `base_height_gaussian` (2.0): exp(-k*(z-z_target)^2) gaussian 형태
- `upright` (2.0): base z축과 world up 벡터의 dot product
- `four_feet_contact` (2.0): 4발 동시 접지 보상

**Negative Penalties:**
- `orientation` (-1.0): roll/pitch 페널티
- `ang_vel_xy` (-0.5): 각속도 페널티
- `low_velocity` (-0.5): 선속도+각속도 페널티 (정지 유도)
- `dof_vel` (-0.001): 관절 속도 페널티
- `action_rate` (-0.01): 액션 변화율 페널티
- `torques` (-0.0002): 토크 페널티
- `collision` (-1.0): 충돌 페널티
- `dof_pos_limits` (-10.0): 관절 한계 초과 페널티

## 결과 요약
| 지표 | Try 1 | Try 2 |
|------|-------|-------|
| Mean Reward | 22.33 | **92.19** |
| Episode Length | 979.5 | **1002.0** (max) |
| base_height_gaussian | - | **1.96 / 2.0** |
| upright | - | **1.79 / 2.0** |
| four_feet_contact | - | **1.65 / 2.0** |
| orientation | -0.007 | -0.008 |
| Noise Std | 0.57 | 0.67 |

## 판정: 성공

- 4발이 모두 바닥에 닿음 (four_feet_contact: 1.65/2.0)
- 수직 자세 잘 유지 (upright: 1.79/2.0)
- 목표 높이 정확히 유지 (base_height_gaussian: 1.96/2.0)
- episode length 최대치 (넘어지지 않음)
- Try 1의 뒷다리 미접지 문제 해결

## 학습 환경
- 1000 iterations, 4096 parallel envs
- 총 학습 시간: ~19분
- 초기 가중치: random (처음부터 학습)
- only_positive_rewards: False (penalty 포함)

## 모델 파일
최종 모델: `legged_gym/logs/quard_stage0/` 내 최신 run
