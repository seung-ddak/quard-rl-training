# walk_8 — Front-Rear Symmetry 보강

## 목적
walk_7 try_2 (mean 164.65) 의 유일한 약점인 **front_rear_stance_symmetry 0.2082**
(left_right 1.0067 대비 1/5 수준) 을 개선.

## walk_7 → walk_8 변화
| 항목 | walk_7 | walk_8 |
|---|---|---|
| front_rear_stance_symmetry scale | 0.25 | **0.60** (2.4x) |
| 그 외 reward | 동일 | 동일 |
| Domain Randomization | friction/push | 동일 |
| resume from | walk_7 try_1 10250 | **walk_7 try_2 11250** |

## 출발점
- load_run: `logs/walk_7/Apr19_15-41-50_walk_7` (try_2 본 학습)
- checkpoint: 11250 (mean 164.65, front_rear_sym 0.2082)

## try_1 계획
- max_iterations 1000 (~40분)
- 합격 기준:
  - front_rear_stance_symmetry **0.50 이상** (2배 이상 개선)
  - mean_reward **155 이상** 유지 (10점 감소까지 허용)
  - left_right_stance_symmetry **0.90 이상** 유지
- 불합격 시:
  - front_rear 개선 미달: scale 추가 상향 (0.90)
  - mean_reward 대폭 저하: scale 소폭 (0.40) 재조정

## 후속 단계 후보
- try_1 OK: 안정화 학습 (1000 iter 추가)
- try_1 실패: scale 조정 후 재학습
- 이후 walk_9에서 lift 돌파 또는 DR 강도 확대
