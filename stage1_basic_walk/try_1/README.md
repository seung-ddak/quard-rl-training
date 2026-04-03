# Stage 1 - Try 1: 평지 걷기 학습

## 결과 요약
| 지표 | Stage 0 최종 | Stage 1 최종 |
|------|-------------|-------------|
| Mean Reward | 22.33 | **38.57** |
| Episode Length | 979.5 | **1002.0** (max) |
| Noise Std | 0.57 | 0.36 |
| tracking_lin_vel | 0.963 | **1.452** |
| tracking_ang_vel | 0.423 | **0.740** |
| feet_air_time | 0.0 | -0.032 |

## 판정: 성공

Stage 0 모델을 이어받아 걷기를 성공적으로 학습했습니다.
- tracking_lin_vel 1.45로 목표 속도를 잘 추적
- tracking_ang_vel 0.74로 회전 명령도 잘 수행
- episode length 최대치 유지 (넘어지지 않음)
- collision 거의 0 (자기 충돌 없음)

## 그래프 설명

### mean_reward.png
- Stage 0 모델(reward ~22)에서 시작하여 ~38까지 상승
- 약 iteration 1300 이후 수렴

### episode_length.png
- 처음부터 거의 max(1002)를 유지
- Stage 0에서 안정적 서기를 배운 덕분

### reward_components.png
- **tracking_lin_vel**: 1.45 (목표 속도 잘 추적)
- **tracking_ang_vel**: 0.74 (회전 명령 수행)
- **feet_air_time**: -0.032 (발을 번갈아 들고 있음)
- **orientation**: -0.012 (수평 유지)
- **dof_pos_limits**: -0.003 (Stage 0 대비 개선)

### noise_std.png
- 0.57에서 0.36으로 감소
- 정책이 안정화됨

## 학습 환경
- 1500 iterations (Stage 0 모델에서 resume, 총 2500)
- 총 학습 시간: ~33분
- 초기 가중치: Stage 0 model_1000.pt
- 자세한 설정: [Stage 1 README](../README.md)

## 모델 파일
최종 모델: `legged_gym/logs/quard_stage1/` 내 최신 run의 `model_2500.pt`
