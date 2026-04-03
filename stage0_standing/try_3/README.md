# Stage 0 - Try 3: 떨림 억제 보상 강화 (실패)

## Try 2 대비 변경사항
| 보상 | Try 2 | Try 3 | 변경 이유 |
|------|-------|-------|-----------|
| four_feet_contact | 2.0 | 3.0 | 4발 접지 강화 |
| ang_vel_xy | -0.5 | -1.0 | 흔들림 억제 |
| low_velocity | -0.5 | -1.0 | 정지 강화 |
| action_rate | -0.01 | -0.05 | 떨림 억제 |
| dof_vel | -0.001 | -0.005 | 관절 떨림 억제 |
| torques | -0.0002 | -0.001 | 토크 절약 |

## 결과 요약
| 지표 | Try 2 | Try 3 |
|------|-------|-------|
| Mean Reward | 92.19 | **5.02** |
| Episode Length | 1002 | **181.8** |
| four_feet_contact | 1.65 | 0.23 |
| upright | 1.79 | 0.24 |
| base_height_gaussian | 1.96 | 0.23 |

## 판정: 실패

페널티를 너무 급격하게 강화하여 학습이 수렴하지 못함.
로봇이 빨리 넘어지며 positive reward를 받을 기회가 줄어듦.
→ Try 4에서 Try 2와 Try 3 사이 값으로 조절.
