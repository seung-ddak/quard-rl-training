# walk_6 try_12 — anti-shuffle 강화 (자율 학습)

## 배경 (try_11 결과)
- mean_reward: **153.92** (목표 170 미달)
- swing_air_height raw: **0.0144** (목표 0.5 미달, lift 사실상 없음)
- no_shuffle 페널티 raw 평균: -0.32 (작은 값, 임계값 10mm 근처에서 머무름)
- 정책이 "lift 10mm 직전" 국부 최적해에 갇힘

## try_12 reward 변경 (4가지)
| 항목 | try_11 | try_12 | 의도 |
|---|---|---|---|
| `swing_air_height` scale | 8.0 | **16.0** | lift 양의 인센티브 두 배 |
| `swing_air_target` | 0.030 | **0.045** | 목표 높이 4.5cm로 상향 |
| `shuffle_lift_thr` | 0.010 | **0.020** | shuffle 임계 2배 (10mm로 부족, 20mm 요구) |
| `no_shuffle` scale | -4.0 | **-6.0** | shuffle 페널티 강화 |

다른 reward 모두 try_11 동일.

## 자율 학습 절차
1. **검증 학습 (300 iter)**: try_11의 9950에서 이어서 → 방향성 확인 (~13분)
2. **합격 기준**:
   - swing_air_height raw 0.014 → **>0.05** (3배 이상 상승)
   - mean_reward 154 → **>140** (큰 하락 없음, 약간 회복도 OK)
3. **합격 시**: max_iterations 1700으로 늘려 본 학습 (try_13 자동 등록)
4. **불합격 시**: scale 또는 target 재조정 후 또다시 300 iter 검증

## 실행 정보
- load_run: walk_6/Apr19_07-45-17_walk_6, checkpoint 9950
- max_iterations: 300 (검증)
- 시작 예정: 2026-04-19 (자동)
