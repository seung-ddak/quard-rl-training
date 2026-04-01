# Stage 0 - Try 3: 균형 잡고 서기

## Try 2 대비 변경점
- default_joint_angles: hip 0.8→**1.0**, knee -1.5→**-1.0** (실제 서있는 자세)
- base_height_target: 0.25→**0.20** (실측 높이 0.22m에 맞춤)
- init pos z: 0.35→**0.30**

## 설정
- only_positive_rewards: False
- tracking_lin_vel: 1.0, tracking_ang_vel: 0.5
- iterations: 1000

## 결과
- 상태: 학습 중
