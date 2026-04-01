# Stage 0 - Try 4: 균형 잡고 서기

## Try 3 대비 변경점
- **j_LR_hip 축 수정**: axis="0 -1 0" → **"0 1 0"** (XML과 일치하도록)
  - 원인: 왼쪽 뒤 다리 hip이 반대로 움직여서 로봇이 즉시 넘어짐
  - 수정 파일: `sim/quard_isaacgym.urdf`

## 설정 (Try 3과 동일)
- default_joint_angles: hip=1.0, knee=-1.0
- base_height_target: 0.20
- init pos z: 0.30
- only_positive_rewards: False
- tracking_lin_vel: 1.0, tracking_ang_vel: 0.5
- iterations: 1000

## 결과
- 상태: 대기
