# walk_8 try_2 — Front-Rear Symmetry 추가 상향

**2026-04-20 학습** · model_13250.pt

## 개입
- front_rear_stance_symmetry scale **0.60 → 0.90** (1.5x 추가 상향)
- 그 외 walk_8 try_1과 동일
- walk_8 try_1 model_12250 (mean 170.12) resume

## 결과 (walk_8 try_1 대비)

| 지표 | try_1 | try_2 | 변화 | 합격선 |
|---|---:|---:|---:|:-:|
| Mean reward | 170.12 | 167.11 | -3.01 | ≥155 ✅ |
| front_rear_sym | 0.5216 | **0.7316** | +0.21 | ≥0.50 ✅ |
| left_right_sym | 0.9757 | 0.9912 | +0.016 | ≥0.90 ✅ |
| swing_height | 1.9543 | 1.9540 | 동일 | ✅ |
| episode_length | 1002 | 998.48 | -3.52 | ✅ |
| tracking_lin_vel | 0.8097 | 0.8493 | +0.04 | ✅ |
| tracking_ang_vel | 0.9378 | 0.8463 | -0.09 | ✅ |
| Value loss | 0.0114 | 2.6128 | ↑ | ⚠️ |

## 수렴 곡선
| iter | reward | 비고 |
|---:|---:|---|
| 200 | 160.45 | 빠른 회복 |
| 500 | 158.93 | dip |
| 800 | 161.72 | 회복 |
| 900 | 165.05 | peak 접근 |
| 950 | 161.82 | dip |
| 990 | 160.06 | dip |
| 997 | 167.11 | 최종 (진동 후) |

**최종 10 iter σ ≈ 3** (try_1의 σ<0.5 대비 6배 변동)

## 해석
- front_rear scale 0.90은 **목표는 달성**했으나 **학습 불안정 초래**
- Oscillation + 높은 value loss → local optimum 간 진동
- Trade-off: front_rear +0.21 / mean -3.01
- 합격 기준 모두 통과, 하지만 try_1이 더 안정적 best

## 다음 단계 후보
1. **walk_8 try_3 scale 0.75**: try_1과 try_2 중간값, 안정성+개선 균형
2. **walk_8 try_2 추가 학습 1000 iter**: 수렴 안정화 시도
3. **walk_9로 진행**: try_1을 최종 walk_8 best로 확정 후 lift/DR 단계

## 현재 best
- **Mean reward 기준**: walk_8 try_1 model_12250 (170.12)
- **Front-rear symmetry 기준**: walk_8 try_2 model_13250 (0.7316)
- 종합 best: 사용자 우선순위에 따라 선택 필요
