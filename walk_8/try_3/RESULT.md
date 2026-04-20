# walk_8 try_3 — Front-Rear Symmetry 중간값 시도

**2026-04-20 학습** · model_13250.pt

## 개입
- front_rear_stance_symmetry scale **0.75** (try_1 0.60 / try_2 0.90 중간값)
- 그 외 walk_8 동일
- walk_8 try_1 model_12250 (안정 base, mean 170.12) resume

## 결과

| 지표 | try_1 | try_2 | try_3 | try_3 평가 |
|---|---:|---:|---:|:-:|
| Mean reward | 170.12 | 167.11 | 165.21 | ❌ 양자 대비 하락 |
| front_rear_sym | 0.5216 | 0.7316 | 0.6075 | △ 중간 |
| left_right_sym | 0.9757 | 0.9912 | 0.9641 | △ 최저 |
| swing_height | 1.9543 | 1.9540 | 1.9537 | ✅ 유지 |
| episode_length | 1002 | 998 | 1002 | ✅ max |
| Value loss | 0.011 | 2.613 | 0.531 | ⚠️ 중간 |
| tracking_lin_vel | 0.810 | 0.849 | 0.864 | ✅ 최고 |

## 수렴 곡선
| iter | reward |
|---:|---:|
| 1 | 11.65 (reset) |
| 200 | 150.28 |
| 500 | 156.53 |
| 800 | **130.85 (dip!)** |
| 950 | 163.13 |
| 987 | 165.21 (final) |

iter 800 대폭 dip 발생 → 완전 수렴 아님.

## 핵심 해석
1. scale 0.75 = "중간값"이지만 **양 극단을 이기지 못함**
   - Mean: try_1 (0.60) > try_3 (0.75) > try_2 (0.90)
   - Sym: try_2 (0.90) > try_3 (0.75) > try_1 (0.60)
2. **monotonic trade-off 확인** — scale 상향할수록 sym↑, mean↓
3. Symmetry 축 튜닝은 포화 근접, **추가 상향 의미 없음**
4. tracking_lin_vel는 try_3이 최고 (0.864) — 특이점이나 종합 부진

## 결론
**walk_8 try_1 model_12250이 종합 best** (mean 170.12 + 안정 수렴 + 합격 기준 전부).

- try_2 front_rear 0.73은 부차적 best (trade-off 비용 큼)
- try_3은 명확한 개선 없음

## 다음 단계
**walk_9로 축 전환**:
- DR 확대 (friction [0.3, 1.7], push 0.5 m/s) — 강건성 축
- 또는 lift 재도전 (walk_6에서 구조적 한계 확인됨, 우선순위 낮음)
- Base: walk_8 try_1 model_12250
