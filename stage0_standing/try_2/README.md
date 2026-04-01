# Stage 0 - Try 2: 균형 잡고 서기 (실패)

## Try 1 대비 변경점
- `only_positive_rewards`: True → **False**
- `tracking_lin_vel`: 0.0 → **1.0**
- `tracking_ang_vel`: 0.0 → **0.5**

## 결과
- **상태**: 실패
- **iterations**: 1000
- **최종 episode length**: 9.8 (최대 68.6)
- **최종 mean reward**: -0.80
- **최종 noise std**: 0.34 (수렴됨)

## 실패 원인
- default_joint_angles(hip=0.8, knee=-1.5)로 서면 로봇이 바닥으로 꺼짐
- 실제 측정 높이: **-0.21m** (바닥 아래!) vs 목표 높이 0.25m
- 초기 자세 자체가 서있을 수 없는 자세

## 발견 사항
- hip=1.0, knee=-1.0 → 높이 0.22m (안정적으로 서있음)
- hip=1.2, knee=-1.0 → 높이 0.35m (높이 서있음)

## 수정 방향 (Try 3)
- default_joint_angles 변경: hip=1.0, knee=-1.0
- base_height_target: 0.25 → 0.20
- init pos z: 0.35 → 0.30

## 생성된 그래프
- `episode_length.png`, `mean_reward.png`, `reward_components.png`, `noise_std.png`
