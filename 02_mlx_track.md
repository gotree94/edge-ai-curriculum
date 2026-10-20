# 02. MLX 병행 트랙 (15%)

대상: MacBook(M시리즈) 학생 + **강사 본인** | 조건: Apple Silicon, macOS
역할: Jetson 이론 구간과 **동시 진행**하는 병행 실습. 하드웨어 추가 비용 0원.

## 설치 (D-1 사전)

```bash
pip install mlx mlx-lm
python -c "import mlx.core as mx; print(mx.metal.is_available())"   # True면 OK
mlx_lm.generate --prompt "hello"      # 기본 모델: mlx-community/Llama-3.2-3B-Instruct-4bit
```

> MLX는 Apple Silicon 전용. Intel Mac·Windows 학생은 이 트랙 제외 → `03_cross_benchmark.md`의 ONNX Runtime(CPU) 경로로 대체.

## L1. 추론·생성 (10분)

```bash
mlx_lm.chat                                   # 챗 REPL
mlx_lm.generate --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
                --prompt "로봇 제어 루프의 주기 요구사항을 설명해줘"
```

- 점검: 4bit 모델이 **내 노트북 RAM** 에서 얼마 만에 몇 토큰/s 나오는지 기록

## L2. 양자화 — Jetson M4 트랙과 개념 동일, 명령만 다름

```bash
# HF 원본 → MLX 4bit 변환·양자화
mlx_lm.convert --model mistralai/Mistral-7B-Instruct-v0.3 -q

# 사전 양자화 모델 직접 사용 (mlx-community, 훨씬 빠름)
mlx_lm.generate --model mlx-community/Llama-3.2-3B-Instruct-4bit --prompt "..."
```

- **개념 매핑표 (필수 회고)**

| Jetson 트랙 | MLX 트랙 |
|---|---|
| FP16/INT8 (TensorRT) | 4bit / 8bit (mlx-lm) |
| `trtexec --fp16/--int8` | `mlx_lm.convert -q` |
| INT8 캘리브레이션 | quantize-method(qmean/qzero) |
| tegrastats 전력 측정 | tokens/s + RAM 사용량 |

- 산출: "같은 모델을 INT8과 4bit로 압축했을 때, 손실 대비 크기·속도 비교" 1쪽

## L3. LoRA 미세튜닝 — Jetson에서 못 하는 것 (강사 포함 학습)

```bash
# train.jsonl: {"prompt": ..., "response": ...} 또는 alpaca 형식
mlx_lm.lora --model mlx-community/Llama-3.2-3B-Instruct-4bit \
            --data train.jsonl \
            --batch-size 4 --lora-layers 16 \
            --num-iterations 200 \
            --adapter-path ./adapters
mlx_lm.generate --model mlx-community/Llama-3.2-3B-Instruct-4bit \
                --adapter-path ./adapters --prompt "..."
```

- 핵심 포인트: **4bit 양자화된 모델 위에** LoRA가 동작 (저사양 하드웨어 = 저비용 파인튜닝 스토리)
- 데이터는 개인 노트북 · 수업용 소형 JSONL로 충분 (수백 샘플)

## L4. 벤치마크 습관 (03_cross_benchmark.md 제출용)

```bash
mlx_lm.benchmark --model mlx-community/Llama-3.2-3B-Instruct-4bit   # 지원 옵션은 --help 확인

# 수동 측정 예시: 토큰/초
python - <<'EOF'
import time
from mlx_lm import load, generate
model, tok = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
p = tok.apply_chat_template([{"role":"user","content":"100자 요약"}], add_generation_prompt=True)
t=time.time(); out=generate(model, tok, prompt=p, max_tokens=256); dt=time.time()-t
print(f"{256/dt:.1f} tok/s, {dt:.2f}s")
EOF
```

- 기록 항목: tok/s, 첫 토큰 지연, RAM 피크, 모델 크기(GB)

## L5. (심화) 서버 모드 — 보드 없이 서비스 배포 체험

```bash
mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit   # OpenAI 호환 엔드포인트
```

- 노트북 = 저가 엣지 서버라는 점을 Jetson Nginx/ROS2 배포와 대비

## 대형 모델 (7B 초과) 주의

- RAM이 모델보다 작으면 느려짐 → macOS 15+: `sudo sysctl iogpu.wired_limit_mb=N` (모델 크기(MB) < N < RAM)
- 수업 시간 내 3B 4bit 권장. 7B는 시연만.

## 강사 팁 (본인 학습 경로 포함)

1. 본인 MacBook(또는 빌려온 M시리즈)으로 **L1→L2→L3을 먼저 직접 돌리기** — Jetson 트랙 준비보다 가벼워 하루면 충분
2. 강사 시연용: 학생 Mac 화면을 공유해 `mlx_lm.convert -q` 전/후 크기 비교를 라이브로 보여주기
3. Jetson 세션 중 MacBook 학생 이탈 방지: L2 매핑표 회고 + L4 제출물을 **동일 양식**으로 취합 → 역할 분리가 아니라 데이터 통합
4. 학생 Mac이 없을 경우: 본인 강사 Mac만으로 시연 + 해당 학생은 ONNX Runtime CPU 경로
