# walk_9 — Domain Randomization 확대

## 목적
walk_8 try_1 (mean 170.12, stable) 위에서 **강건성 확대**.
- walk_7/8 DR: friction [0.5, 1.5], push 0.3 m/s / 8s interval
- walk_9: friction [0.3, 1.7], push 0.5 m/s / 6s interval (강도 및 빈도 증가)

## walk_8 try_1 → walk_9 변화
| 항목 | walk_8 try_1 | walk_9 |
|---|---|---|
| friction_range | [0.5, 1.5] | **[0.3, 1.7]** |
| max_push_vel_xy | 0.3 | **0.5** |
| push_interval_s | 8.0 | **6.0** |
| front_rear scale | 0.60 | 0.60 (유지) |
| 그 외 reward/env | 동일 | 동일 |

## 출발점
- load_run: `logs/walk_8/Apr20_06-26-10_walk_8` (walk_8 try_1)
- checkpoint: 12250 (mean 170.12, front_rear 0.52)

## try_1 합격 기준
- mean_reward **≥ 155** (DR 강화 15점 감소 허용)
- front_rear_stance_symmetry **≥ 0.40** (유지)
- left_right_stance_symmetry **≥ 0.85** (약간 저하 허용)
- episode_length **≥ 900** (push 대응 실패 많아질 수 있음)

## 실패 시 후속
- DR 너무 과한 경우: friction [0.4, 1.6], push 0.4로 축소
- Gait 파괴: push_interval 다시 8s로 완화
