# Stage 0 - Try 5: 균형 잡고 서기

## Try 4 대비 변경점
- **init pos z**: 0.30 → **0.22** (목표 높이 0.20에 가깝게, 낙하 충격 최소화)
- **damping (kd)**: 4.0/3.0 → **8.0/6.0** (댐핑 비율 개선, 관절 진동 억제)
- **collision 보상**: -1.0 → **-0.1** (바닥 접촉 페널티 완화, 서기에 필수)
- **dof_pos_limits 보상**: -10.0 → **-0.5** (무릎 구부리기 허용, 안정적 자세 유도)

## 변경 이유
Try 1~4까지 episode length ~9.5로 로봇이 즉시 넘어짐.
원인 분석 결과 URDF가 아닌 **보상 함수 충돌 + PD 게인 언더댐핑**이 근본 원인:
1. dof_pos_limits=-10.0이 무릎 굽히기를 과도하게 억제
2. kd=4.0으로 댐핑 비율 0.45 (임계댐핑 1.0 필요)
3. 초기 높이 0.30이 목표 0.20보다 10cm 높아 착지 시 불안정
4. collision=-1.0이 바닥 접촉(서기에 필수)을 억제

## 설정
- default_joint_angles: hip=1.0, knee=-1.0 (Try 4와 동일)
- base_height_target: 0.20 (동일)
- stiffness (kp): 200 (동일)
- only_positive_rewards: False
- iterations: 1000

## 결과
- **상태**: 학습 중
- Mean reward: -
- Episode length: -

## 이전 시도 요약

| Try | 결과 | 원인 |
|---|---|---|
| Try 1 | 실패 | only_positive_rewards=True → 학습 신호 없음 |
| Try 2 | 실패 | default_joint_angles가 서있을 수 없는 자세 |
| Try 3 | 중단 | j_LR_hip 축 반대 (URDF 버그) |
| Try 4 | 실패 | 축 수정만으로 부족, 보상/게인 문제 |
| Try 5 | 진행중 | 보상 함수 + 댐핑 수정 |
