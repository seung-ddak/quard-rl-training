# Stage 1: 평지 걷기 학습

## 목표
Stage 0에서 학습한 서기 자세를 기반으로, 평평한 지면 위에서 전진/후진/좌우 이동 및 회전을 학습합니다.

## Stage 0 대비 변경사항
| 파라미터 | Stage 0 | Stage 1 | 변경 이유 |
|---------|---------|---------|-----------|
| lin_vel_x | [0, 0] | [-0.3, 0.5] | 전진/후진 명령 |
| lin_vel_y | [0, 0] | [-0.3, 0.3] | 좌우 이동 명령 |
| ang_vel_yaw | [0, 0] | [-0.5, 0.5] | 회전 명령 |
| tracking_lin_vel | 1.0 | 1.5 | 속도 추적 강화 |
| tracking_ang_vel | 0.5 | 0.8 | 회전 추적 강화 |
| feet_air_time | 0.0 | 1.0 | 걸음 패턴 유도 |
| max_iterations | 1000 | 1500 | 더 복잡한 task |
| initial weights | random | Stage 0 model_1000 | 전이 학습 |

## 보상 함수 설정
| 보상 항목 | 가중치 | 설명 |
|-----------|--------|------|
| tracking_lin_vel | 1.5 | 목표 선속도 추적 |
| tracking_ang_vel | 0.8 | 목표 각속도 추적 |
| feet_air_time | 1.0 | 발이 번갈아 들리도록 유도 |
| lin_vel_z | -2.0 | 수직 속도 억제 |
| orientation | -1.0 | 수평 자세 유지 |
| base_height | -0.5 | 목표 높이 유지 |
| collision | -1.0 | 충돌 페널티 |
| dof_pos_limits | -10.0 | 관절 한계 초과 페널티 |

## Try 기록
각 시도별 결과는 하위 폴더에 저장됩니다.
