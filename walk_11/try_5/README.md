# walk_11 try_5 — soft thigh penalty scale 완화 + clean baseline

> 노트: 원래 try_4 로 작성했으나 train.py auto-increment 가 try_5 로 저장 (try_4 폴더 미리 생성한 게 원인). 내용 동일.

## 사용자 지시 (2026-04-27)

> "But without that, my robot walking using thigh"
> "yes proceeding now"

try_2 (hard term only, soft penalty 없음) 에서 ep_len 780 / rew 30 까지 회복했음에도 육안으로 thigh-walking 잔존 → soft penalty 필요. 단 try_3 의 `collision=-10` + walk_10 try_21 baseline 조합은 학습 붕괴 (rew 3~12, noise std 0.83 폭주).

## 진단 요약

| 메커니즘 | force 임계값 | 출처 |
|---|---|---|
| Termination | `> 1.0 N` | `legged_robot.py:301-302` |
| Collision (soft penalty) | `> 0.1 N` | `legged_robot.py:1259` |

→ thigh 가 0.1~1.0 N 으로 살짝 닿는 grazing 영역은 hard term 으로 못 잡음. soft penalty 필수.
단 `collision=-10` 은 walk_5/10 시절 `penalize_contacts_on=[]` (= 항상 0) 일 때 의미 없던 값. thigh 추가하는 순간 갑자기 활성화된 미튜닝 페널티.

## 변경

### 1. `QuardWalk11Cfg.rewards.scales` 추가
```python
class rewards(QuardWalk10Cfg.rewards):
    class scales(QuardWalk10Cfg.rewards.scales):
        collision = -2.0   # -10 → -2 (PPO 압도 방지, hard term -6 보다 작게)
```

### 2. baseline 교체
```python
load_run = ".../walk_10/Apr26_14-31-05_walk_10"   # try_17 (clean trot)
checkpoint = 9950
```
(try_3 의 Apr27_03-53-43_walk_10 try_21 = thigh-walking 박힌 baseline 폐기)

### 3. 유지 (try_3 와 동일)
- `penalize_contacts_on = ["thigh"]` (soft)
- `terminate_after_contacts_on = ["base_link", "thigh"]` (hard)
- terrain: rough_slope, DR, commands, entropy_coef = 0.008
- max_iterations = 2000

## 학습 설정

- baseline: walk_10 try_17 model_9950 (Apr26_14-31-05)
- 학습 타겟 iter: 9950 → 11950

## 평가 기준

- ep_len 600+ 안정 (transition shock 후 회복)
- noise std < 0.6 유지 (try_3 의 0.83 폭주 방지)
- play.py 육안: thigh **brushing 도 없음**, calf 만 짚는 깔끔한 trot
- diagonal_gait 1.0+ 유지

## 가설

clean trot baseline (try_17) 에서 출발 + 부드러운 thigh penalty (-2) → 정책이 thigh 회피 방향으로 점진 수정. -10 처럼 noise 폭발 없이 안정 수렴 기대.

## 결과 (2026-04-27 학습 완료)

| 항목 | 값 | 평가 |
|---|---|---|
| 총 iteration | **11949 / 11950** (정상 종료) | ✅ |
| 학습 시간 | **9620초 (160분 ≈ 2.67시간)** | — |
| 총 timesteps | 196,608,000 | — |
| Mean reward (peak iter 10549) | **28.91** | ✅ try_2 (30) 수준 근접 |
| Mean reward (final iter 11949) | **24.59** | ⚠️ peak 대비 -15% 회귀 |
| ep_len (peak iter 10549) | **720.17** | ✅ 600+ 충족 |
| ep_len (final) | **647.69** | ✅ 600+ 충족 |
| noise std (final) | **0.63** | ⚠️ 평가기준 0.6 살짝 초과 |
| diagonal_gait (final) | **0.99** | ✅ 1.0 근접 |
| diagonal_propulsion (final) | 0.82 | ✅ |
| feet_slip | -0.99 | — |
| **collision (thigh 접촉)** | **-0.0013** | ✅ try_3 (-0.009) 의 14% (thigh grazing 거의 사라짐) |
| terrain_level | 0.0000 | ⚠️ 진급 X (walk_10 try_21 와 동일 한계) |
| 최종 ckpt | `logs/walk_11/Apr27_14-09-56_walk_11/model_11950.pt` | — |
| Peak ckpt 후보 | `model_10550.pt` (rew 28.9, ep 720) | 육안 비교 필요 |

### 학습 진행 (요약)

- iter 9950 (start): rew -0.18, ep_len 22 (예상된 transition shock — thigh termination + soft penalty 첫 노출)
- iter ~10000: 빠른 회복, rew 16, ep 398, noise 0.29
- **iter 10549 peak**: rew 28.9, ep_len 720, diag_gait 1.13 ← clean walking 정점
- iter 10949 dip: rew 15.5, ep_len 408, diag_gait 0.65 (regression)
- iter 11149~11949: rew 15~24, ep_len 440~647 oscillation, 천천히 회복
- iter 11949 final: rew 24.59, ep_len 647, diag_gait 0.99, noise 0.63

### try_3 대비 효과

| 항목 | try_3 (실패) | try_5 (현재) | 개선 |
|---|---|---|---|
| Mean reward 말기 | 3~12 | 17~24 | ✅ 2~3배 |
| ep_len 말기 | 200~437 | 512~647 | ✅ +50% |
| noise std | 0.57→0.83 폭주 | 0.29→0.63 안정 | ✅ |
| collision | -0.0090 | -0.0013 | ✅ 86% ↓ |

## 분석

### 좋은 점
- **soft penalty + clean baseline 조합 작동**: collision -10 → -2 변경이 PPO noise 폭주 막음
- thigh 접촉 -0.0013 수준 = grazing 거의 제거 (사용자 의도 달성)
- diag_gait 0.99 = trot 패턴 유지

### 한계 / 의문
- **iter 10549 peak 후 regression** (rew 28.9 → 15.5 → 24.6 oscillation): noise std 가 0.29 → 0.63 까지 올라간 것이 원인 가능성. entropy_coef 0.008 이 후반엔 과한 듯
- **terrain_level = 0**: walk_10 try_21 와 동일 — 평지 row 0 에서 reward max 받는 게 진급보다 유리한 구조 (별개 문제)
- final (model_11950) 보다 peak (model_10550) 가 더 좋은 정책일 가능성 → play.py 두 모델 비교 필요

## 다음 단계 후보

- **play.py 육안**: model_10550 (peak) vs model_11950 (final). thigh 사용 여부 + trot 안정성 확인
- 만약 model_10550 이 best → real_baseline 갱신 후 다음 stage
- 만약 둘 다 thigh 잔존 → collision scale -2 → -3 추가 강화 (try_6) 또는 entropy_coef 낮춤 (0.008 → 0.005)

