# walk_6 : 품질 재설계 단계 (Quality Redesign — re-defined 2026-04-18)

## 변경 사유 (2026-04-18 재정의)
원래 walk_6는 "domain randomization 강건성 단계"였다. 그러나 walk_5 try_12 (iter
7148/7150 plateau) 결과 다음이 미해결로 남았다:

1. **발 높이 (foot clearance)** — `rew_swing_height` raw ≈ 0.279에서 정체
   (target 0.040m, worst foot effective ≈ 0). control penalty와 trade-off에
   걸려 정책이 더 이상 발을 들지 않음.
2. **대각쌍 추진 비대칭** — 한 대각쌍은 들고 차서 전진을 만들고, 반대 대각쌍은
   바닥을 살짝 건드리기만 함 (stuck_foot_penalty=0이라 영구 stance는 아님,
   다만 추진력 없음).
3. **직진 deviation 페널티 부재** — walk_4의 `left_right_stance_symmetry`는
   *어깨 자세 대칭*일 뿐, 로봇 위치가 직선에서 벗어나는 것에 대한 페널티
   아님. 평지에서는 통하지만 지형 randomization 도입 시 무용지물.

품질이 미흡한 정책 위에 DR을 얹으면 결함이 더 굳어진다. 따라서 **walk_6는
DR 도입 직전 품질 재설계 단계로 재정의**한다. DR은 walk_7로 미룬다.

walk_5 try_13 (병행 진행 중)이 plateau를 깨면 walk_6는 try_13 결과를 받아
재설계 reward로 마무리; 깨지 못하면 walk_6에서 신규 reward 함수까지 도입.

## walk_6 도입 목표

### A. 직진 deviation 페널티 신규
- **신규 함수 `_reward_straight_line_deviation`**:
  spawn 시점 대비 누적 lateral position drift.
  `‖(p_xy - p_xy_spawn) ⋅ ⊥(forward_cmd)‖²`
  명령이 순수 +x 직진이면 결국 |y - y_spawn|² 와 동일.
- 이유: 속도(lin_vel_y) 페널티는 순간 noise에 약하고, 방향 drift 누적은
  지형/외란에서 로봇이 서서히 휘어 나가는 진짜 실패 모드.
- 활성화 후 `tracking_lin_vel`/`tracking_ang_vel` 약간 완화.

### B. 대각쌍 추진력 명시 보상
- **신규 함수 `_reward_diagonal_propulsion`** (또는 기존 `diagonal_gait` 강화):
  대각쌍 (LF+RR), (RF+LR) 각각이 stance 구간에서 **순방향 지면반력**을
  생성했는지 측정. 한쪽 쌍이 scuff만 하면 직접 페널티.
  구현 후보:
    - 각 발 stance 시 `foot_velocity_x` 부호와 base 전진 속도 일치 여부
    - 또는 stance 시간 중 양 쌍의 평균 z-force 균형
- diagonal_gait은 *위상*만 보고 *추진*은 안 본다 → 핵심 결함 지점.

### C. swing height reward 재설계 (try_13 결과 따라)
- try_13가 swing_height raw ≥ 0.5 달성하면 그대로 walk_6에 가져옴
- 못 미치면: per-foot **mean** 모드와 min 모드 가중 평균 (예: 0.5×min + 0.5×mean)
  → min만으로는 worst foot 한 발만 펌핑해도 안 풀림, mean과 섞으면 4발 평균
  높이도 보상

### D. (옵션) Domain Randomization 보수적 도입
walk_7로 미루는 것이 기본 방침. 단, walk_5 try_13가 매우 잘 되면 walk_6에
*가벼운* friction randomization (0.8~1.1) 정도만 추가 후 walk_7에서 본격화.

## 새로 추가될 reward 함수 (구현 예정)

### `_reward_straight_line_deviation`
```python
def _reward_straight_line_deviation(self):
    move_cmd = (self.commands[:, 0] > self.min_command_norm).float()
    # spawn 위치는 reset 시 self._spawn_xy 버퍼에 저장
    # 명령 forward 방향(world frame): yaw 명령이 0이라 spawn 시점 forward 사용
    delta_xy = self.root_states[:, :2] - self._spawn_xy
    forward = self._spawn_forward  # (N, 2)
    perp = torch.stack([-forward[:, 1], forward[:, 0]], dim=1)
    lateral = torch.sum(delta_xy * perp, dim=1)
    sigma = getattr(self.cfg.rewards, "straight_dev_sigma", 0.30)
    return -move_cmd * torch.square(lateral) / (sigma * sigma)
```

### `_reward_diagonal_propulsion` (스케치)
```python
def _reward_diagonal_propulsion(self):
    # 양 대각쌍의 stance 시 z-force 비율 균형 측정
    move_cmd = (self.commands[:, 0] > self.min_command_norm).float()
    f = self.contact_forces[:, self.feet_indices, 2].clamp(min=0.0)
    pair_a = f[:, 0] + f[:, 3]   # LF + RR
    pair_b = f[:, 1] + f[:, 2]   # RF + LR
    total = (pair_a + pair_b).clamp(min=1e-6)
    ratio = pair_a / total
    # 시간 평균(ema)으로 비교, 이상 = 0.5
    self._pair_force_ema = 0.98 * self._pair_force_ema + 0.02 * ratio
    sigma = getattr(self.cfg.rewards, "pair_force_sigma", 0.10)
    return move_cmd * torch.exp(-torch.square(self._pair_force_ema - 0.5) / (2.0 * sigma * sigma))
```
(상세 구현은 try_13 결과 검토 후 확정)

## 버퍼 추가 필요
- `self._spawn_xy` (N, 2): reset 시 root_states[:, :2] 복사
- `self._spawn_forward` (N, 2): reset 시 quat → yaw → forward unit vec
- `self._pair_force_ema` (N,): 0.5로 초기화

## Resume
- `load_run`: walk_5 **try_13 최종 checkpoint** (try_13 결과 후 갱신)
- 임시 placeholder: walk_4 final `Apr14_16-06-11_walk_4` ckpt 5950 (try_13와 동일 시작점)

## 실행 (try_13 완료 후)
```bash
LD_LIBRARY_PATH=/home/xiangyue/.conda/envs/isaacgym/lib \
/home/xiangyue/.conda/envs/isaacgym/bin/python \
legged_gym/scripts/train.py --task quard_walk_6 --headless
```

## walk_6 try_1 모니터 포인트 (예정)
- [ ] `rew_straight_line_deviation` |raw| < 0.5 (lateral drift < 0.3m)
- [ ] `rew_diagonal_propulsion` > 0.7 (양 쌍 추진 균형)
- [ ] `rew_swing_height` raw ≥ 0.5
- [ ] `rew_tracking_lin_vel` > 1.0 유지
- [ ] play.py 육안: 직선 이탈 0.5m 이내 / 양 대각쌍 모두 들고 차는 동작

## walk_7 (예정)
walk_6 품질 확보 후 본격 domain randomization (마찰, mass, push, 지형).
