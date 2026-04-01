# Stage 0 - Try 1: 균형 잡고 서기 (실패)

## 결과
- **상태**: 실패
- **iterations**: 1000
- **최종 episode length**: 37.8 (최대 408.8)
- **최종 noise std**: 7.52 (발산)
- **최종 mean reward**: 0.00

## 실패 원인
- `only_positive_rewards = True` 설정 + 모든 보상이 음수(페널티)
- 총 보상이 항상 0으로 클리핑됨 → PPO가 학습 신호를 받지 못함
- noise std가 1.0 → 7.5로 계속 증가 (발산)

## 설정
- URDF: `sim/quard_isaacgym.urdf`
- 지형: 평면
- 속도 명령: 없음 (0, 0, 0)
- only_positive_rewards: True (문제)
- tracking_lin_vel/ang_vel: 0.0 (양수 보상 없음)

## 수정 방향 (Try 2)
- `only_positive_rewards = False`로 변경
- 또는 서있기 보상을 양수로 추가

## 생성된 그래프
- `episode_length.png` — 에피소드 길이 변화
- `mean_reward.png` — 평균 보상
- `reward_components.png` — 개별 보상 항목
- `noise_std.png` — 탐색 노이즈 변화
