# walk_6 Final — try_11 9950 확정

**Final checkpoint**: `logs/walk_6/Apr19_07-45-17_walk_6/model_9950.pt`
**Backup**: `training_record/walk_6/try_11/model_9950.pt`

## 최종 성적
- mean_reward: **153.92**
- tracking_lin_vel: 0.78
- diagonal_gait: 0.60
- left_right_stance_symmetry: 0.92
- swing_air_height (buggy min-ref): 0.0144
- swing_air_height (fixed stance-ref, try_16): 0.0055 (**실제 lift ≈ 5mm**)

## try_12~17 실패 기록
| try | 변경 | 결과 | 결론 |
|---|---|---|---|
| 12 | shuffle_thr 0.010→0.020, no_shuffle -4→-6 | mean 127, deviation 악화 | 가혹 페널티는 gait 파괴 |
| 14 | swing_air_height 8→16, target 0.030→0.045 | mean 102, deviation -2.48 | scale만 올리면 weaving |
| 15 | 속도↓ + entropy↑ + scale 12 | mean 86, symmetry 0.21 | 다각 튜닝은 안 먹힘 |
| 16 | **ground_ref bug fix** (min → stance-mean) | mean 125, swing 0.0055 | **실제 lift 정체 폭로** |
| 17 | ground_ref fix + 1000 iter | mean 135, swing 0.0057 | 정책 적응해도 lift 그대로 |

## 핵심 발견
**`_reward_swing_air_height`와 `_reward_no_shuffle`의 `ground_ref = min(foot_z)` 버그** — 정책이 한쪽 발을 바닥 아래로 밀어 min을 낮추고 다른 발의 상대 lift를 과장하는 속임수를 학습. `legged_robot.py`에서 stance 평균 기반으로 수정 완료 (try_16 이후 유지).

**이 로봇의 실제 lift 한계 ≈ 5-10mm.** reward 조정만으로는 돌파 불가. 3kg 소형 사족보행 로봇의 모터 토크/질량 비율 한계로 판단.

## walk_7로 이월
- 추가 lift 개선은 walk_6 reward 내에서 불가능
- 다음 방향 후보: DR(friction+push) 강건성 / swing_cycle_peak 재활성 / foot_stance_max_time 커리큘럼
