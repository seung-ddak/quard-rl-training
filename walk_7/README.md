# walk_7 — Domain Randomization 단계

## 목적
walk_6 try_11 final (mean 154) 정책 위에서 **강건성** 학습.
lift 개선은 walk_6에서 구조적 한계 확인 (5mm), walk_7은 다른 축.

## walk_6 → walk_7 변화
| 항목 | walk_6 | walk_7 |
|---|---|---|
| randomize_friction | False | **True** [0.5, 1.5] |
| push_robots | False | **True**, interval 8s, max 0.3 m/s |
| reward | try_11 final | 동일 |
| 속도 범위 | 0.28~0.50 m/s | 동일 |

## 출발점
- load_run: `logs/walk_6/Apr19_07-45-17_walk_6` (try_11)
- checkpoint: 9950 (mean 154, symmetry 0.92)

## try_1 검증 학습
- max_iterations 300 (~13분)
- 합격 기준: mean_reward 유지 (>130), symmetry 유지 (>0.8)
- 불합격: DR 강도 완화 (friction [0.7, 1.3], push_vel 0.2)

## 차후 단계
- try_1 OK → try_2에서 본 학습 (2000 iter)
- lift 개선은 walk_8에서 다른 접근 (swing_cycle_peak 재활성, stance time curriculum 등)
