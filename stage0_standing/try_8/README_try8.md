# Stage 0 Standing - Try8
## Try 4 모델 기반 추가 학습 (Resume Fine-tuning)

---

## 개요

| 항목 | 내용 |
|------|------|
| **학습 방식** | Try 4 최종 모델(model_1000.pt)에서 이어서 학습 |
| **목표** | Try 4의 안정적인 서기 자세를 기반으로 4발 접지 완성도 향상 |
| **총 추가 학습** | 1000 iterations (Try 4 1000 + Try 8 1000 = 누적 2000 iter) |
| **상태** | 진행 중 / 완료 |

---

## 왜 Try 4에서 이어서 학습했는가

### Try 4 성과 (베이스라인)

Try 4는 Stage 0 전체 시도 중 가장 안정적인 결과를 달성했다.

| 지표 | Try 4 결과 | 평가 |
|------|-----------|------|
| Mean Reward | 104.20 | Try 1~6 중 최고 |
| Episode Length | 1002.0 (최대) | 한 번도 넘어지지 않음 |
| four_feet_contact | 2.22 / 2.5 (89%) | 거의 4발 접지 |
| base_height_gaussian | 2.00 / 2.0 (100%) | 목표 높이 완벽 유지 |
| upright | 1.83 / 2.0 (92%) | 수직 자세 안정 |
| Noise Std | 0.33 | 정책 안정화 (낮을수록 좋음) |

### Try 5, 6, 7 (처음부터 학습) 실패 원인 분석

| Try | 문제 |
|-----|------|
| Try 5 | PD 게인 과도 증가로 관절 진동 |
| Try 6 | stand_still 보상이 지배 → 쓰러진 채로 관절만 유지하는 나쁜 전략 학습 |
| Try 7 (처음부터) | 위 문제들 수정했으나 처음부터 학습 시 수렴 불안정 |

→ **결론: Try 4의 학습된 가중치를 재활용하는 것이 가장 효율적**

---

## Try 7 변경 사항 (Try 4 reward 대비)

### 핵심 변경 4가지

#### [CRITICAL] 1. four_feet_contact 강화
```
Try 4: four_feet_contact = 2.5
Try 7: four_feet_contact = 6.0  (+140%)
```
**이유:** Try 4에서 4발 접지 89%까지 달성했으나 완전한 100% 미달.  
보상을 크게 높여 나머지 11%를 채우도록 유도.

#### [CRITICAL] 2. stand_still 보상 축소
```
Try 4: stand_still = 0.0 (비활성)
Try 6: stand_still = 1.0 (과도하게 활성화 → 실패)
Try 7: stand_still = 0.3 (적절한 수준으로 활성화)
```
**이유:** Try 6에서 stand_still이 지배 보상이 되어 쓰러진 채로 관절만 유지하는  
나쁜 전략이 학습됨. 0.3으로 낮춰 보조 역할만 하도록 제한.

#### [CRITICAL] 3. dof_pos_limits 페널티 강화
```
Try 4: dof_pos_limits = -10.0
Try 7: dof_pos_limits = -20.0  (2배 강화)
```
**이유:** Try 6 분석에서 관절 한계 초과가 -3.21로 심각한 수준.  
Try 4에서도 잠재적으로 관절이 한계 근처에서 동작하고 있을 가능성 차단.

#### [MEDIUM] 4. collision 페널티 강화
```
Try 4: collision = -1.0
Try 7: collision = -2.0  (2배 강화)
```
**이유:** 몸통이 바닥에 닿는 상황을 더 강하게 억제.  
Try 4에서 이미 안정적이지만 추가 안전 마진 확보.

### 전체 보상 비교표

| 보상 항목 | Try 4 | Try 7 | 변경 |
|-----------|-------|-------|------|
| base_height_gaussian | 2.0 | 3.0 | ↑ |
| upright | 2.0 | 3.0 | ↑ |
| four_feet_contact | 2.5 | **6.0** | ↑↑ (핵심) |
| stand_still | 0.0 | **0.3** | 신규 활성화 |
| orientation | -1.0 | -2.0 | ↑ |
| ang_vel_xy | -0.7 | -1.0 | ↑ |
| low_velocity | -0.7 | -1.0 | ↑ |
| dof_vel | -0.003 | -0.02 | ↑ |
| action_rate | -0.03 | -0.05 | ↑ |
| torques | -0.0005 | -0.001 | ↑ |
| collision | -1.0 | **-2.0** | ↑↑ |
| dof_pos_limits | -10.0 | **-20.0** | ↑↑ |
| feet_force_symmetry | 0.0 | -0.5 | 신규 |
| feet_symmetry | 0.0 | -0.3 | 신규 |

---

## 학습 설정

### 환경 설정
```
num_envs       : 4096
num_obs        : 48
num_actions    : 12
terrain        : flat plane
episode_length : 20s (max 1002 steps)
```

### 제어 설정
```
control_type   : P (Position control)
Kp (stiffness) : 40.0 (shoulder / hip / knee)
Kd (damping)   : 1.5  (shoulder / hip / knee)
action_scale   : 0.25
decimation     : 4
```

### PPO 학습 설정
```
max_iterations : 1000
resume         : True
load_run       : <Try 4 폴더명>
checkpoint     : 1000
entropy_coef   : 0.01
```

### 초기 자세 (default_joint_angles)
```
shoulder : 0.0 rad (모든 다리)
hip      : 1.0 rad (모든 다리)
knee     :-1.0 rad (모든 다리)
init_pos : [0, 0, 0.22m]
```

---



## 성공 판단 기준

| 지표 | 목표값 | 근거 |
|------|--------|------|
| four_feet_contact | > 5.5 / 6.0 (92%) | Try 4 대비 향상 |
| base_height_gaussian | > 2.8 / 3.0 (93%) | Try 4 수준 유지 |
| upright | > 2.7 / 3.0 (90%) | Try 4 수준 유지 |
| collision | > -0.5 | Try 4보다 개선 |
| dof_pos_limits | > -1.0 | 관절 초과 억제 |
| Episode Length | 1002.0 (최대) | 넘어지지 않음 |
| Mean Reward | > 104.20 | Try 4 초과 |

---

## 이전 시도 전체 이력

| Try | 방식 | Mean Reward | Ep Length | 판정 | 주요 원인 |
|-----|------|-------------|-----------|------|-----------|
| Try 1 | 처음부터 | 22.33 | 979.5 | 실패 | 기본 설정, 보상 미조정 |
| Try 2 | 처음부터 | 92.19 | 1002.0 | 부분 성공 | 보상 체계 개선 |
| Try 3 | 처음부터 | 95.02 | 181.8 | 실패 | 페널티 과도 → 바로 넘어짐 |
| Try 4 | 처음부터 | **104.20** | **1002.0** | **최고** | Try 2/3 중간값으로 균형 |
| Try 5 | 처음부터 | - | - | 실패 | PD 게인 과도 증가 |
| Try 6 | 처음부터 | 113.95 | 1002.0 | 실패(시각) | stand_still 지배 → 쓰러진 채 버팀 |


---

## 다음 단계 계획

Try 8 성공 시 → **Stage 1 (평지 걷기)** 진입
- Try 8 model_1000.pt를 Stage 1 초기 모델로 사용
- tracking_lin_vel, tracking_ang_vel 보상 활성화
- feet_air_time 보상 추가 (발을 들어올리는 걷기 동작 유도)
- 속도 명령 범위: lin_vel_x [-0.3, 0.5], ang_vel_yaw [-0.5, 0.5]
