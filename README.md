# AutoCIM-Agent

**CIM(Compute-In-Memory) 반도체를 위한 LangGraph 기반 6-에이전트 HW-SW 공동설계 최적화 프레임워크**

레이어별 양자화 비트/컬럼 프루닝 비율과 크로스바 하드웨어 제약(ADC/DAC 해상도, IR-drop, 소자 노이즈)을 함께 고려해, 정확도·에너지·지연시간의 Pareto 최적점을 자율적으로 탐색하는 폐루프(closed-loop) 에이전트 파이프라인입니다. 배경과 전체 설계 의도는 [`docs/research_plan.md`](docs/research_plan.md)를, 코드 작성 규칙은 [`CLAUDE.md`](CLAUDE.md)를 참고하세요.

## 아키텍처

```mermaid
graph TD
    START[__start__] --> planner["@planner (탐색/대리모델)"]
    planner --> tuner["@tuner (PyTorch QAT)"]

    tuner --> mapper["@mapper (NoC 매핑)"]
    tuner --> profiler["@profiler (물리 시뮬레이션)"]

    mapper --> verifier["@verifier (IR-drop/노이즈 검증)"]
    profiler --> verifier

    verifier --> evaluator["@evaluator (수렴 판정/Pareto rank)"]

    evaluator -- "Re-plan with History" --> planner
    evaluator -- "HITL Interrupt" --> hitl["hitl_human_approval"]
    evaluator -- "Converged (Done)" --> END[__end__]

    hitl --> planner
```

| 에이전트 | 역할 | 실제 구현 여부 |
|---|---|---|
| `@planner` | LHS 워밍업 + IDW 대리모델/UCB 탐색(stage별 독립, 최대 40차원), LLM 도구 호출로 최종 확정 | 실제 |
| `@tuner` | 크로스바 정렬 fake-quantization + 구조적 컬럼 프루닝, 실제 QAT fine-tuning (`tools/qat.py`) | 실제 (CIFAR10 서브셋, CUDA/MPS/CPU 자동 감지) |
| `@mapper` / `@profiler` | NoC 홉카운트/대역폭 매핑, ADC/DAC/크로스바 에너지·지연 물리 시뮬레이션 | 실제 (fast-approximation) |
| `@verifier` | IR-drop/노이즈 마진 기반 수렴 판정 | 실제 |
| `@evaluator` | Pydantic 스키마 검증, NSGA-II 스타일 Pareto rank, 재탐색/HITL 라우팅 | 실제 |

> **스코프 한계 (의도적으로 문서화됨):** `@mapper`/`@profiler`는 NeuroSim/CIM-Loop 같은 정밀 시뮬레이터가 아니라 물리 법칙 기반의 fast-approximation이며(`calibration_factors`로 보정), `@tuner`의 학습 데이터는 CIFAR10 512/128/1000장 서브셋입니다. 후보 간 **상대 비교**에는 신뢰할 수 있지만, 절대 수치를 그대로 벤치마크/제품 스펙으로 사용하지 마세요.

## 주요 기능
- **로컬 우선 LLM**: `@planner`는 기본적으로 로컬 Ollama 모델(`ollama:qwen2.5:7b`)을 사용 -- 하드웨어 스펙이 외부로 나가지 않음. `AUTOCIM_PLANNER_MODEL`로 클라우드 provider를 opt-in 가능
- **세션 영속성**: `SqliteSaver` 기반 체크포인트로 프로세스 재시작 후에도 `--thread-id`로 재개
- **HITL(Human-in-the-Loop)**: 수렴 정체 시 연구원 개입 요청 (dynamic `interrupt()`)
- **LLM 호출 운영**: 재시도/지수 백오프, rate-limit 대응, 토큰/비용 추적, 누적 비용·토큰 상한(kill-switch)
- **구조화된 관측성**: iteration별 JSON Lines 로그(`observability.py`) + 후보/Pareto front 이동을 보여주는 HTML 대시보드(`tools/dashboard.py`)
- **GPU 자동 감지**: `tools/qat.py`가 CUDA → Apple MPS → CPU 순으로 자동 감지 (`AUTOCIM_QAT_DEVICE`로 override 가능)
- **병렬 워밍업**: LHS 워밍업 후보들은 서로 독립적이므로, `--parallel-warmup-workers`로 스레드 풀에서 동시 평가 가능 (`tools/batch_warmup.py`)

## 설치

가상환경 사용을 권장합니다 (전역 site-packages와의 충돌 방지 -- 아래 "문제 해결" 참고):

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows; macOS/Linux는 source .venv/bin/activate
pip install -r requirements.txt
```

새 venv에서 `pip install -r requirements.txt` 후 `pytest`까지 정상 통과하는 걸 확인했습니다 (170개 테스트).

`torch`/`torchvision`은 버전만 고정돼 있고 실제 wheel(CPU 전용 vs CUDA)은 어느 index에서 설치하느냐로 결정됩니다:

```bash
# CPU 전용 (다운로드 용량 최소화 -- CI/Docker가 이 방식 사용)
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# GPU(CUDA) 사용 -- 드라이버가 지원하는 CUDA 버전에 맞는 index 선택
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

설치 후 `python -c "import torch; print(torch.cuda.is_available())"`로 GPU 인식 여부를 확인할 수 있습니다.

## 실행

`@planner`(llm.py)는 기본적으로 **로컬 Ollama 모델**(`ollama:qwen2.5:7b`)을 사용합니다 -- 하드웨어 스펙/후보 설정을 외부 클라우드로 보내지 않아도 되도록, IP에 민감한 배포(반도체 회사 등)를 염두에 둔 기본값입니다. 처음 실행 전 한 번만:

```bash
# https://ollama.com 에서 Ollama 설치 후
ollama pull qwen2.5:7b
ollama serve   # 보통 설치 시 자동으로 백그라운드 서비스로 등록됨

python main.py --model-id resnet18 --dashboard-out report.html
```

클라우드 provider를 쓰고 싶으면 `AUTOCIM_PLANNER_MODEL`을 설정하면 됩니다 (`langchain.chat_models.init_chat_model` 스펙 문자열). `.env.example`을 `.env.local`로 복사해 값을 채우면 -- `main.py`가 실행할 때마다 자동으로 읽어들이므로 (import 시점이 아니라 `python main.py` 실행 시점에만; gitignore됨) -- 터미널을 새로 열 때마다 다시 `export`할 필요가 없습니다:

```bash
cp .env.example .env.local
# .env.local을 열어 AUTOCIM_PLANNER_MODEL과 해당 provider의 API 키를 채우기

python main.py --model-id resnet18 --dashboard-out report.html
```

또는 기존처럼 셸에서 직접 `export`해도 동일하게 동작합니다:

```bash
export AUTOCIM_PLANNER_MODEL="anthropic:claude-sonnet-4-5-20250929"
export ANTHROPIC_API_KEY="..."

python main.py --model-id resnet18 --dashboard-out report.html
```

`@planner`는 `PlannerLayerDecision`을 forced tool-calling(`tool_choice=<정확한 도구 이름>`)으로 호출하는데, 이걸 실제로 지원하지 않거나 필수 필드를 빠뜨리는 provider/모델도 있습니다. 아래는 실제로 end-to-end(실제 resnet18/mobilenet_v2+CIFAR10 QAT 포함, 10~20-stage `layer_configs` 리스트 전부)로 검증된 조합입니다:

| Provider | 모델 | 비고 |
|---|---|---|
| Ollama (로컬, 기본값) | `ollama:qwen2.5:7b` | resnet18(10 stage)/mobilenet_v2(20 stage) 전부 검증됨. `qwen2.5:14b`는 이 프로젝트의 forced tool-calling 프롬프트에서 오히려 빈 응답을 자주 반환해 비권장 |
| Anthropic | `anthropic:claude-sonnet-4-5-20250929` | 클라우드 중 기본 권장 |
| Groq (무료 티어 가능) | `groq:llama-3.3-70b-versatile` | `llama-3.1-8b-instant`는 `LayerBitConfig`의 필수 필드(`activation_bits`)를 종종 빠뜨려 실패함 -- 더 작은 모델로 바꾸기 전에 스키마 준수를 직접 확인할 것 |
| Google Gemini (무료 티어 가능) | `google_genai:gemini-3.5-flash-lite` | 스키마 준수 확인됨, 토큰 소모 적음. `gemini-3.5-flash`도 되지만 내부 reasoning 토큰 때문에 같은 요청에 ~2배 더 많은 토큰을 씀 |

Groq/Gemini를 쓰려면 `pip install -r requirements.txt`에 이미 포함된 `langchain-groq`/`langchain-google-genai`가 필요하고, 각각 `GROQ_API_KEY`/`GOOGLE_API_KEY` 환경변수를 설정하면 됩니다.

**`--hw-config`를 생략하고 터미널에서 직접 실행하면**, 매번 JSON 파일 경로를 손으로 쓰는 대신 대화형으로 물어봅니다: (1) 커스텀 값 직접 입력 (2) 저장된 예시 파일(`examples/hw_configs/`) 중 선택 (3) 기본값 사용. 직접 입력 시 각 필드에서 `b`(또는 `back`/`뒤로`)를 입력하면 바로 앞 필드로 돌아가 다시 입력할 수 있고(맨 첫 필드에서 입력하면 [1]/[2]/[3] 메뉴로 돌아감), 스키마 검증에 실패해도(예: `noc_topology`에 오타) 이미 입력한 다른 필드는 그대로 유지된 채 문제된 필드만 원래 기본값으로 재설정됩니다. 입력한 스펙이 `tools/calibration.py`의 레퍼런스와 정확히 일치하지 않으면, 그 자리에서 가장 가까운 레퍼런스로 근사 보정할지(`--allow-approximate-calibration`과 동일 효과) 바로 물어봅니다. `--hw-config`를 명시하거나 스크립트/CI처럼 터미널이 아닌 환경에서 실행하면 이 프롬프트는 뜨지 않고 기존과 동일하게 동작합니다.

> **GPU + `--parallel-warmup-workers` 병용 시 주의**: GPU가 하나뿐이면 워밍업 후보들이 전부 같은 GPU 메모리를 동시에 두고 경쟁합니다. OOM이 나면 `--parallel-warmup-workers`를 낮추거나, 이 단계만 `AUTOCIM_QAT_DEVICE=cpu`로 강제하세요.

주요 CLI 옵션 (`main.py`):

| 옵션 | 설명 |
|---|---|
| `--model-id` | 대상 모델 (`resnet18`, `mobilenet_v2`, `vit_tiny`) |
| `--hw-config PATH` | `HWConfig` 필드를 담은 JSON 파일 (생략 시 내장 샘플 스펙 사용). 예시는 [`examples/hw_configs/`](examples/hw_configs/) 참고 |
| `--thread-id ID` | 세션 재개용 LangGraph thread id |
| `--checkpoint-db PATH` | 체크포인트 SQLite 경로 (`:memory:`로 영속성 끄기 가능) |
| `--dashboard-out PATH` | 세션이 멈추거나 끝날 때마다 HTML 대시보드를 이 경로에 기록 |
| `--list-sessions` | `--checkpoint-db`에 저장된 모든 세션(thread_id, 상태, iteration_count)을 나열하고 종료 |
| `--parallel-warmup-workers N` | (opt-in) 새 세션 시작 시 LHS 워밍업 후보들을 순차 그래프 루프 진입 전에 N개 스레드로 동시 평가 (`tools/batch_warmup.py`). 생략 시 기존처럼 순차 실행 |
| `--target-accuracy FLOAT` | (opt-in) 이 정확도 이상이어야 "수렴(Converged)"으로 인정 -- `@verifier`의 IR-drop/노이즈 검사만으로는 실제 정확도가 낮아도 수렴 처리될 수 있어서 추가된 게이트. 생략 시 기존처럼 정확도는 게이트하지 않음 (`nodes/evaluator.py`) |
| `--target-energy-pj FLOAT` | (opt-in) 이 값 이하여야 수렴으로 인정 |
| `--target-latency-ms FLOAT` | (opt-in) 이 값 이하여야 수렴으로 인정 |
| `--allow-approximate-calibration` | (opt-in) `--hw-config`가 `tools/calibration.py`의 `KNOWN_REFERENCES`와 정확히 일치하지 않을 때, 가장 가까운 레퍼런스의 보정 계수를 스케일링 로우 기반으로 근사 적용 (`bootstrap_approximate_calibration_factors`). 생략 시 기존처럼 정확히 일치하지 않는 하드웨어는 미보정(factor=1.0) 상태로 남음. 근사 적용 여부와 매칭된 레퍼런스/거리는 세션 시작 시 `[calibration] ...` 로그로 출력됨 |

> **`--target-*` 미달 시 동작**: 기존 `MAX_RETRY_LIMIT`(3회) 재시도 → HITL 흐름을 그대로 재사용합니다. 목표 미달 후보도 `candidate_history`에는 정상 기록됩니다(surrogate 학습에 유효한 데이터이므로) -- 다만 `is_converged=False`로 남아 재탐색을 계속 유도합니다.

### 주요 환경변수

| 변수 | 용도 |
|---|---|
| `AUTOCIM_PLANNER_MODEL` | (옵션) LLM provider:model 스펙 -- 생략 시 로컬 `ollama:qwen2.5:7b` 사용 |
| `AUTOCIM_LLM_MAX_RETRIES` / `_BASE_DELAY_SECONDS` / `_MAX_DELAY_SECONDS` | LLM 호출 재시도/백오프 튜닝 |
| `AUTOCIM_LLM_COST_PER_1K_INPUT_USD` / `_OUTPUT_USD` | 토큰당 비용 추정 활성화 (미설정 시 비용은 항상 `null`) |
| `AUTOCIM_LLM_MAX_TOTAL_COST_USD` / `AUTOCIM_LLM_MAX_TOTAL_TOKENS` | 런 전체 누적 상한 도달 시 LLM 호출 자체를 중단 |
| `AUTOCIM_LOG_DIR` | 구조화 로그(JSONL) 저장 위치 (기본 `.cache/logs/`) |
| `AUTOCIM_QAT_TRAIN_SIZE` / `_VAL_SIZE` / `_TEST_SIZE` / `_BATCH_SIZE` | QAT fine-tuning/평가 데이터셋 크기 (기본 512/128/1000/32 — CIFAR10 데모 서브셋). `TRAIN_SIZE`/`TEST_SIZE`는 `full`로 설정하면 해당 split 전체 사용 (wall-clock 증가) |
| `AUTOCIM_QAT_SEED` | QAT train/val/test 분할 시드 (기본 0) |
| `AUTOCIM_QAT_DEVICE` | QAT 학습에 사용할 디바이스 (`cuda`, `mps`, `cpu`, `cuda:0` 등). 미설정 시 CUDA → MPS → CPU 순으로 자동 감지 |

## Docker

```bash
docker build -t autocim-agent .
```

`.cache/`(체크포인트 DB, 다운로드된 모델 가중치/CIFAR10)를 마운트하지 않으면 `--rm`으로 컨테이너가 사라질 때 전부 같이 사라집니다 -- 매 실행마다 다시 다운로드하고, `--thread-id` 세션 재개도 안 됩니다. `--dashboard-out`으로 쓰는 파일도 마찬가지로 마운트해야 컨테이너 밖에서 열어볼 수 있습니다:

```bash
docker run --rm \
  --env-file .env.local \
  -v "$(pwd)/.cache:/app/.cache" \
  -v "$(pwd)/report.html:/app/report.html" \
  autocim-agent --model-id resnet18 --dashboard-out report.html
```

(`--env-file .env.local`은 `-e AUTOCIM_PLANNER_MODEL=... -e ANTHROPIC_API_KEY=...`를 하나씩 나열하는 대신 `.env.local` 내용을 그대로 전달합니다.)

> **로컬 Ollama 기본값과 Docker**: 컨테이너 안에서는 `ollama:qwen2.5:7b` 기본값이 가리키는 `localhost:11434`가 컨테이너 자신을 가리켜 호스트의 Ollama에 닿지 않습니다. `-e OLLAMA_HOST=http://host.docker.internal:11434`로 호스트의 Ollama를 가리키거나, `.env.local`에서 `AUTOCIM_PLANNER_MODEL`을 클라우드 provider로 오버라이드하세요.

## 테스트

```bash
pytest
```

전부 stub 백엔드(가짜 LLM, 합성 모델/데이터)로 동작해 네트워크/실제 학습 없이 수 초 내 완료됩니다. `tests/conftest.py`의 autouse fixture가 실제 QAT 학습·LLM API 호출을 자동으로 가짜로 대체합니다. `.github/workflows/ci.yml`이 push/PR마다 동일하게 실행합니다.

## 문제 해결

**`ImportError`/`numpy._core._multiarray_umath` 관련 에러, 또는 엉뚱한 패키지 버전이 로드됨**
`PYTHONPATH` 환경변수가 이 프로젝트와 무관한 다른 프로젝트의 venv를 가리키고 있을 가능성이 높습니다. `PYTHONPATH`는 **가상환경을 새로 만들어도 사라지지 않고 그대로 상속**되며, venv 자체의 site-packages보다 먼저 검색되기 때문에 다른 프로젝트의 패키지가 이 프로젝트의 것보다 우선 로드될 수 있습니다.

```bash
echo $PYTHONPATH        # 값이 있고, 이 프로젝트와 무관해 보인다면 원인일 가능성이 높음
PYTHONPATH= pytest      # 임시로 비우고 재실행해서 확인
```

원인이 맞다면 셸 프로필(`.bashrc`/`.zshrc`/Windows 환경변수)에서 `PYTHONPATH`를 제거하거나, 이 프로젝트를 실행할 때마다 `PYTHONPATH=` 로 비워서 실행하세요. CI(`ci.yml`)와 이 README의 예시 명령어들은 전부 이렇게 명시적으로 비운 상태로 실행됩니다.

## 프로젝트 구조

```
graph.py, main.py, state.py, llm.py, middleware.py,
observability.py, store.py   # 코어 오케스트레이션/런타임 (LangGraph 조립, 상태 스키마, LLM 운영, 구조화 로깅)
nodes/                        # 6개 에이전트 노드 함수
tools/                        # 백엔드 시뮬레이터 (QAT, 물리 시뮬레이션, 탐색 알고리즘, 병렬 워밍업, 대시보드)
schemas/                      # Pydantic 입출력 스키마
tests/                        # pytest 스위트 (전부 stub 기반)
docs/                         # 연구 계획서 등 부가 문서
examples/hw_configs/          # --hw-config 예시 JSON
```

## 라이선스

[BSD-3-Clause](LICENSE). 이 프로젝트를 인용할 때는 [`CITATION.cff`](CITATION.cff)를 참고하세요.
