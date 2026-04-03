# Stage 0 - Try 1: 안정적인 서기 자세 학습

## 결과 요약
| 지표 | 초기값 | 최종값 |
|------|--------|--------|
| Mean Reward | 0.01 | **22.33** |
| Episode Length | 12.5 | **979.5** (max ~1000) |
| Noise Std | 1.00 | 0.57 |
| tracking_lin_vel | 0.005 | **0.963** |
| tracking_ang_vel | 0.001 | **0.423** |

## 판정: 성공

로봇이 거의 최대 에피소드 길이(~1000 steps)까지 넘어지지 않고 서 있습니다.
tracking_lin_vel이 0.96으로 수렴하여 제자리에 안정적으로 서 있는 것을 확인했습니다.

## 그래프 설명

### mean_reward.png
- 총 보상이 0에서 약 22까지 꾸준히 상승
- 약 400 iteration 이후 수렴

### episode_length.png
- 에피소드 길이가 빠르게 max(~1000)에 도달
- 약 200 iteration부터 안정적으로 최대치 유지

### reward_components.png
- **tracking_lin_vel**: 0.96 (제자리 유지 성공)
- **tracking_ang_vel**: 0.42 (회전 없이 유지)
- **orientation**: -0.007 (거의 수평 유지)
- **base_height**: -0.002 (목표 높이 0.20m 유지)
- **torques**: -0.039 (적절한 토크 사용)
- **dof_pos_limits**: -0.014 (관절 한계 약간 초과 - 개선 여지)

### noise_std.png
- 탐색 노이즈가 1.0에서 0.57로 감소
- 정책이 안정화되며 탐색량 줄어듦

## 학습 환경
- 1000 iterations, 4096 parallel envs
- 총 학습 시간: ~27분
- GPU: cuda:0, PhysX
- 자세한 설정: [Stage 0 README](../README.md)

## 모델 파일
최종 모델: `legged_gym/logs/quard_stage0/Apr03_14-24-44_/model_1000.pt`
