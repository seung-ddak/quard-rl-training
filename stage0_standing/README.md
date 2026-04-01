# Stage 0: 균형 잡고 서기

## 목표
- 평면에서 넘어지지 않고 **제자리에 서있기**
- 이동 명령 없이 기본 자세 유지
- 이후 보행 학습의 기반이 되는 단계

## 로봇 정보
- URDF: `RL/sim/quard.urdf`
- 총 질량: 2.966 kg
- DOF: 12 (4다리 x 3관절)
- Effort limit: 20 Nm

## 환경 설정
| 항목 | 값 |
|---|---|
| 지형 | 평면 (plane) |
| 환경 수 | 4096 |
| 속도 명령 | 없음 (0) |
| 노이즈 | 없음 |
| 도메인 랜덤 | 없음 |
| PD 게인 | kp=200, kd=4.0 (shoulder/hip), kd=3.0 (knee) |
| action_scale | 0.25 |
| base_height_target | 0.25 m |

## 주요 보상 함수
| 보상 | 가중치 | 설명 |
|---|---|---|
| orientation | -2.0 | 몸체 수평 유지 (핵심) |
| base_height | -1.0 | 목표 높이 유지 (핵심) |
| lin_vel_z | -2.0 | 수직 속도 억제 |
| ang_vel_xy | -0.1 | 좌우 흔들림 억제 |
| torques | -0.0002 | 에너지 절약 |
| action_rate | -0.01 | 부드러운 동작 |
| dof_pos_limits | -10.0 | 관절 한계 초과 방지 |
| stand_still | -0.5 | 제자리 유지 |

## 실행 방법
```bash
export LD_LIBRARY_PATH=/home/xiangyue/.conda/envs/isaacgym/lib:$LD_LIBRARY_PATH
cd /mnt/gstore/home/xiangyue/RL
/home/xiangyue/.conda/envs/isaacgym/bin/python legged_gym/legged_gym/scripts/train.py --task quard_stage0
```

## 성공 기준
- 20초 에피소드 동안 넘어지지 않음
- 몸체가 수평 유지 (roll, pitch < 0.1 rad)
- 목표 높이(0.25m) 근처 유지

## 학습 결과
- 상태: 미진행
- iterations: -
- 비고: -
