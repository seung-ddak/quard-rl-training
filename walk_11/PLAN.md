# walk_11 try_1 — calf-only contact (thigh termination)

## 사용자 지시 (2026-04-27)

> "we have to retrain with walk_11 base line Apr26_14-31-05_walk_10. now my robot using thigh. Include finish option if touched thigh, everything without calf you know what i mean?"

walk_10 try_21 (model_11950) play.py 육안 확인 결과: **thigh 가 ground 에 닿는 자세**. 굴곡 적응 과정에서 정책이 thigh 로 짚는 편법 학습. calf 만 ground 접촉하도록 강제.

## 변경

### 1. 새 walk class: `QuardWalk11Cfg(QuardWalk10Cfg)`

asset class override:
```python
class asset(QuardWalk10Cfg.asset):
    penalize_contacts_on = []                                   # termination 으로 충분
    terminate_after_contacts_on = ["base_link", "thigh", "shoulder"]
```

기존 `["base_link"]` 만 termination → `base_link + thigh + shoulder` 모두 termination.
calf (= foot) 와 base 는 ground 접촉 가능, **thigh/shoulder 접촉 즉시 episode 종료**.

### 2. `QuardWalk11CfgPPO(QuardWalk10CfgPPO)`

```python
class runner(QuardWalk10CfgPPO.runner):
    run_name = "walk_11"
    experiment_name = "walk_11"
    max_iterations = 2000
    resume = True
    load_run = "/mnt/gstore/home/xiangyue/RL/legged_gym/logs/walk_10/Apr26_14-31-05_walk_10"
    checkpoint = 9950   # walk_10 try_17 (자연 보행 + flat heightfield 적응)
```

### 3. 유지 (walk_10 와 동일)

- terrain: rough_slope (proportions=[0,1,0,0,0]), random amp = 0.02 + 0.08*difficulty
- Reward set: walk_5 회귀 (try_17 와 동일)
- DR: friction [0.45, 1.35], push_robots = True
- commands.lin_vel_x: [0.24, 0.42]
- num_rows=8, num_cols=8, max_init_terrain_level=0, curriculum=True
- entropy_coef = 0.008

### 4. Task registration

- `legged_gym/envs/__init__.py`: `quard_walk_11` 등록
- `train.py` TASK_WALK_MAP: walk_11 추가
- `play.py` TASK_ALIASES: walk_11 / quard_walk_11 추가

## 학습 설정

- baseline: walk_10 try_17 model_9950 (clean policy on smooth pyramid heightfield)
- 학습 타겟 iter: 9950 → 11950

## 평가 기준

- ep_len: 처음에 큰 drop 예상 (thigh termination 즉시 ep_len ↓), 1000 iter 내 회복
- terrain_level: 평지 (row 0) 안정 보행 + 회복
- play.py 육안: **thigh 가 ground 에 닿지 않음**, calf 만 짚는 깔끔한 trot
- 만약 ep_len 회복 못 하고 collapse → max_iterations 늘리거나 termination 완화

## 가설

baseline (try_17, model_9950) = 평지에서 자연 trot 보행 정책. 새 termination 규칙 + 굴곡 환경 동시 노출 → 처음엔 collapse, 점차 thigh 사용 안 하는 방향으로 정책 수정.

## try_1 결과 (2026-04-27 중단)

- iter 9950 → 10800 (850 iter, 67분) 후 정지
- ep_len: 78 → peak 722 → **회귀 394~484**
- Mean reward: 1.94 → peak 21 → **회귀 14**
- diagonal_gait/propulsion 모두 ↓
- **원인**: thigh+shoulder 양쪽 termination 이 너무 strict, 정책이 안정 해법 못 찾음

## try_2 변경 (2026-04-27)

`terminate_after_contacts_on = ["base_link", "thigh"]`   ← shoulder 제거

이유: shoulder 닿으면 사실상 base_link 도 닿아서 중복 신호. thigh 만 명확히 막으면 사용자 의도 (thigh-walking 제거) 달성.

baseline 동일 (walk_10 try_17 model_9950), max_iterations 2000, entropy 0.008 그대로.

## try_2 결과

(학습 완료 후 try_2/README.md 로 옮기며 채움)

## try_6 — sharpening on try_5 peak (2026-04-28)

try_5 결과: peak iter 10549 (rew 28.91, ep_len 720) 후 regression (rew 24.59, noise std 0.63).
원인 추정: entropy_coef 0.008 이 후반엔 과한 탐색, trot 어색함 잔존.

### 변경 (`quard_config.py:559-580`)

| 항목 | try_5 | try_6 |
|---|---|---|
| baseline | walk_10 try_17 model_9950 | **walk_11 try_5 model_10550** (peak) |
| entropy_coef | 0.008 | **0.005** (noise 진정, policy sharpening) |
| diagonal_gait | 2.0 | **2.5** (어색한 trot 회복 +25%) |
| max_iterations | 2000 | **1000** (sharpen 만, 과학습 방지) |
| collision | -2.0 | -2.0 (유지) |
| terminate_after_contacts_on | base+thigh | (유지) |

### 학습 타겟

10550 → 11550 (1000 iter, 약 80분 예상)

### 평가 기준

- ep_len 700+ 안정 (peak 회복 + 유지)
- noise std < 0.5 (sharpening)
- play.py 육안: trot 더 깔끔, thigh grazing 0
- diagonal_gait ≥ 1.0

### 가설

clean trot peak 에서 출발 + entropy 낮춤 + diagonal_gait reward ↑ → policy 가 안정 trot 으로 수렴 (탐색 줄이고 활용 강화).
