# Stage 0: 안정적인 서기 자세 학습

## 목표
4족 로봇(Quard)이 평평한 지면 위에서 넘어지지 않고 안정적으로 서 있는 자세를 학습합니다.

## 로봇 사양 (새 모델 - 2026-04-03)
- **총 질량**: ~3.93 kg
- **몸체(base_link)**: 0.959 kg
- **다리 구조**: shoulder → thigh → thigh_pully(fixed) → calf (4다리)
- **Actuated joints**: 12개 (shoulder, hip, knee x 4)
- **서보 토크**: HTD-85H, ±5.5 N·m
- **URDF**: `sim/quard_rl.urdf`

## 학습 환경 설정
| 파라미터 | 값 | 설명 |
|---------|-----|------|
| num_envs | 4096 | 병렬 환경 수 |
| terrain | plane | 평평한 지면 |
| init_pos | [0, 0, 0.22] | 초기 높이 22cm |
| control_type | P (Position) | 위치 제어 |
| stiffness | 25.0 (all joints) | PD 제어 강성 |
| damping | 0.8 (all joints) | PD 제어 댐핑 |
| action_scale | 0.25 | 액션 스케일 |
| decimation | 4 | 제어 주기 (4 sim steps) |
| max_iterations | 1000 | 최대 학습 반복 |

## 보상 함수 설정
| 보상 항목 | 가중치 | 설명 |
|-----------|--------|------|
| tracking_lin_vel | 1.0 | 선속도 추적 (명령=0) |
| tracking_ang_vel | 0.5 | 각속도 추적 (명령=0) |
| lin_vel_z | -2.0 | 수직 속도 억제 |
| ang_vel_xy | -0.05 | 롤/피치 각속도 억제 |
| orientation | -1.0 | 수평 자세 유지 |
| base_height | -0.5 | 목표 높이(0.20m) 유지 |
| torques | -0.0002 | 토크 사용 최소화 |
| collision | -1.0 | 충돌 페널티 |
| action_rate | -0.01 | 급격한 액션 변화 억제 |
| dof_pos_limits | -10.0 | 관절 한계 초과 강한 페널티 |

## 기타 설정
- Domain Randomization: OFF
- Noise: OFF
- Push robots: OFF
- 이동 명령: 없음 (서기만 학습)

## Try 기록
각 시도별 결과는 하위 폴더에 저장됩니다.
