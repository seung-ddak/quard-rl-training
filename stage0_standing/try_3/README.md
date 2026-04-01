# Stage 0 - Try 3: 균형 잡고 서기 (중단)

## Try 2 대비 변경점
- default_joint_angles: hip 0.8→1.0, knee -1.5→-1.0 (실제 서있는 자세)
- base_height_target: 0.25→0.20 (실측 높이 0.22m에 맞춤)
- init pos z: 0.35→0.30

## 결과
- **상태**: 중단 (~467/1000 iterations)
- **Mean reward**: -0.60
- **Episode length**: 9.3 (여전히 짧음)
- **noise std**: 0.32

## 중단 사유
- j_LR_hip 관절 축이 XML과 반대 방향인 것을 발견
- URDF: axis="0 -1 0" vs XML(정상): axis="0 1 0"
- 왼쪽 뒤 다리가 반대로 움직여서 즉시 넘어짐
- Try 4에서 축 수정 후 재학습
