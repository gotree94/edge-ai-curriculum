# 03. 교차 벤치마크 랩 (15%)

대상: 전원 (조별) | 목적: **"저가 하드웨어에서 얼마나 다른가"** 를 숫자로 증명하고 팀별 산출물 통합

## 배포 매트릭스 (핵심 산출물)

같은 모델을 아래 경로에 배포하고 표를 채운다.

| 하드웨어 | 추론 엔진 | 비용 | 측정값 |
|---|---|---|---|
| Jetson Orin Nano | TensorRT FP16 / INT8 | ~50만원 | mAP, FPS, P50/P99, W |
| Jetson Nano (구형) | TensorRT FP16 / INT8 | ~10만원(중고) | 동일 |
| MacBook (M시리즈) | MLX 4bit 또는 ONNX Runtime(MPS) | 개인 노트북 | mAP/정확도, tok/s·FPS, RAM |
| Raspberry Pi 5 (있을 시) | ONNX Runtime CPU | ~60달러 | 동일 |
| 학생 노트북 (Windows/Linux) | ONNX Runtime CPU | 개인 노트북 | 동일 |

**한 줄 요약 표시어**: "정확도 98% 유지에 드는 비용·전력은?" — 이 숫자가 과제 전체의 결론.

## 실습 A. CV 모델 교차 (20명 공통)

1. 공통 ONNX 파일(`model_fp16.onnx`, opset 13)을 **전원이 동일 경로에서 다운로드**
2. 각 환경에서 추론 100회 반복 → P50/P99, FPS, 메모리
3. 정확도는 동일 검증셋 100장으로 통일 (팀 간 비교 가능하게)
4. 결과를 팀별 1행씩 매트릭스에 기록 → 강사가 전체 취합

```bash
# Jetson (01트랙)
trtexec --loadEngine=model_fp16.engine --nbRuns=100

# Mac / PC (02트랙, ONNX Runtime 경로)
pip install onnxruntime
python bench_ort.py --model model_fp16.onnx --runs 100   # 과제 제공 스크립트
```

## 실습 B. LLM 교차 (Booth: 소수 팀)

- Jetson Nano 4GB로는 3B급 LLM 부적합 → **Nano는 제외**, Orin vs MacBook 비교
- MacBook: `mlx_lm.benchmark` (02트랙 L4)
- Orin: llama.cpp GGUF 또는 TensorRT-LLM (강사가 사전 사전 준비해 둔 경로 선택 — 준비 부담 주의, 시연용으로 권장)

> 주의: CV 실습 A가 공통 과제, LLM 실습 B는 선택. 시간 초과 시 A만으로 수료 가능.

## 측정 규칙 (전원 동일 — 편차 방지)

1. 워밍업 10회 후 측정 100회
2. 전력: Jetson은 `sudo tegrastats`의 VDD_IN 평균, Mac은 `sudo powermetrics --samplers power` (또는 전체 와트 미측정 시 명시)
3. 열 상태: 측정 전 2분 idle로 온도 안정화, 열저하 시 재측정
4. **P99를 반드시 기록** — 평균만으로는 로봇 실시간성 얘기가 안 됨 (제어 루프 30Hz 예산 = P99 < 33ms 관점)

## 제출 양식 (팀 1종)

```
팀명 / 모델명 / 검셋
| HW | 엔진 | 크기(MB) | mAP또는정확도 | FPS | P50 | P99 | W | 비고 |
|----|------|----------|----------------|-----|-----|-----|---|------|
```

### 회고 질문 3개 (개인 제출)

1. 저가 Nano에서 INT8이 필요한 이유를 전력 관점에서 설명
2. MacBook 4bit와 Jetson INT8은 "다른 양자화"인가 같은가 — 공통점·차이점 3줄
3. 목표 FPS를 달성 못 한 하드웨어에서 모델을 어떻게 줄이겠는가 (증류·프루닝·해상도 중 택1, 근거)

## 강사 팁

- 사전에 자기 표본 데이터 1행을 완성해 두고 시작 (빈 표를 보고 시작하면 팀이 표본 오류를 베낀다)
- Pi 5가 없으면 학생 Windows 노트북 ONNX CPU 행으로 대체 — 저가 축은 Nano가 이미 담당
- 취합은 스프레드시트 링크 1개로 실시간 공유 → 세션 말미에 전체 표를 스크린으로 공개하는 것이 동기부여 효과 큼
