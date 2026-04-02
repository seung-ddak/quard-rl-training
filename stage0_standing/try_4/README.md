# Stage 0 - Try 4: 균형 잡고 서기

## Try 3 대비 변경점
- **j_LR_hip 축 수정**: axis="0 -1 0" → **"0 1 0"** (XML과 일치하도록)
  - 원인: 왼쪽 뒤 다리 hip이 반대로 움직여서 로봇이 즉시 넘어짐
  - 수정 파일: `sim/quard_isaacgym.urdf` (수정 완료)

## 설정 (Try 3과 동일)
- default_joint_angles: hip=1.0, knee=-1.0
- base_height_target: 0.20
- init pos z: 0.30
- only_positive_rewards: False
- tracking_lin_vel: 1.0, tracking_ang_vel: 0.5
- iterations: 1000

## To-Do List

- [ ] Try 4 학습 실행
  ```bash
  export LD_LIBRARY_PATH=/home/xiangyue/.conda/envs/isaacgym/lib:$LD_LIBRARY_PATH
  /home/xiangyue/.conda/envs/isaacgym/bin/python legged_gym/legged_gym/scripts/train.py --task quard_stage0 --headless
  ```
- [ ] 학습 완료 후 그래프 생성
  ```bash
  /home/xiangyue/.conda/envs/isaacgym/bin/python plot_training.py --log <로그파일> --outdir training_record/stage0_standing/try_4 --name "Stage 0 - Standing (Try 4)"
  ```
- [ ] play.py로 로봇 움직임 확인
  ```bash
  /home/xiangyue/.conda/envs/isaacgym/bin/python legged_gym/legged_gym/scripts/play.py --task quard_stage0
  ```
- [ ] episode length가 충분히 길면 (>500) → Stage 1으로 진행
- [ ] episode length가 짧으면 → 보상/PD 게인 조정 후 Try 5
- [ ] 결과 GitHub push
- [ ] GitHub 토큰 폐기 후 재발급 (보안)
  - https://github.com/settings/tokens 에서 현재 토큰 삭제

## 이전 시도 요약

| Try | 결과 | 원인 |
|---|---|---|
| Try 1 | 실패 | only_positive_rewards=True → 학습 신호 없음 |
| Try 2 | 실패 | default_joint_angles가 서있을 수 없는 자세 |
| Try 3 | 중단 | j_LR_hip 축 반대 (URDF 버그) |
| Try 4 | 실패 (reward -0.59, ep_len 9.56) | 축 수정만으로 부족, 보상/댐핑 문제 |
