# 01. Jetson 메인 트랙 (70%)

대상: 전원 | 보드: Nano(학생) + Orin(강사 시연)

## M1. 환경과 생태계 이해 (이론, 노트북)

- Jetson 스택: L4T → CUDA → cuDNN → TensorRT → JetPack (버전이 **묶여** 있음)
- 구형 Nano = JetPack 4.6.x / Orin = JetPack 6.x → 학생 보드와 강사 보드의 툴 버전이 다름을 먼저 설명
- 실습: `cat /etc/nv_tegra_release`, `dpkg -l | grep nvidia-jetpack` 로 각 보드 스택 확인

## M2. PyTorch → ONNX 내보내기 (노트북, 전원 동시)

```python
import torch
model.eval()
dummy = torch.randn(1, 3, 640, 640)
torch.onnx.export(model, dummy, "model.onnx",
                  opset_version=13,           # TensorRT 8(Nano) 호환을 위해 낮게 유지
                  input_names=["images"], output_names=["output"])
```

- 점검: `onnx.checker.check_model`, Netron으로 그래프 육안 확인
- 함정 포인트: 최신 opset / 커스텀 연산은 Nano(TensorRT 8)에서 폴백 → **NMS는 엔진 밖**에서 처리하는 패턴으로 통일

## M3. TensorRT 빌드 (보드)

```bash
# FP16
trtexec --onnx=model.onnx --saveEngine=model_fp16.engine --fp16

# 추론 시간·throughput 베이스라인 측정
trtexec --loadEngine=model_fp16.engine --nbRuns=100 --dumpProfile
```

- INT8: calibration 필요 → `polygraphy`로 캘리브레이션 캐시 생성 후 `--int8 --calib=calib.cache`
- 강사 시연은 Orin(TensorRT 10), 학생 실습은 Nano(TensorRT 8)로 **명령 옵션 차이**를 그대로 목격시키기

## M4. 양자화 실습 — ROI가 가장 높은 세션

| 단계 | 체크포인트 |
|---|---|
| FP32 → FP16 | 메모리 절반, mAP 손실 ≈ 0, FPS 1.5~2배 |
| FP16 → INT8 (PTQ) | 캘리브레이션 데이터 100~500장, mAP 손실 <1pt 목표 |
| QAT (선택) | PTQ로 손실이 2pt 넘을 때만, 시간 부족 시 시연만 |

- 산출: mAP·FPS·메모리 3열 표를 팀마다 반드시 기록

## M5. 프로파일링과 전력 측정 (보드 필수 습관)

```bash
sudo tegrastats        # Jetson 전체 전력·클럭·온도 (VDD_IN)
# 또는 jtop (pip install jtop) — 온도/팩트 표시 편리
```

- 기록 규칙: **P50/P99 지연시간, FPS, W, mAP** 를 매번 같은 하네스로
- 열저하시 FPS 급락 확인 → 방열的重要性을 숫자로 학습

## M6. 증류 프로젝트 (개인 과제)

1. Teacher: GPU에서 학습한 중간형 모델 (예: YOLOv8m)
2. Student: n 사이즈 — soft label/output distillation로 학습
3. Orin/Nano에서 배포: "Teacher 대비 FPS ×N, mAP 손실 <1~2pt" 스타그래프 제출

## M7. 저가 대조군 — Nano의 역할 전환

- 팀별로 **동일 ONNX 모델**을 Nano(저가)와 Orin(표준)에 동시 배포
- 결과는 03_cross_benchmark.md 매트릭스로 취합

## 시간 배분 (예시, 총 6세션)

| 세션 | 내용 | 보드 사용 |
|---|---|---|
| 1 | M1 + M2 | 불필요 (노트북) |
| 2 | M3 TensorRT 빌드 | 필수 |
| 3 | M4 양자화 | 필수 |
| 4 | M5 프로파일링 | 필수 |
| 5 | M7 대조군 + 매트릭스 취합 | 필수 |
| 6 | M6 증류 결과 발표 | 불필요 (발표) |

## 강사 팁

- Nano는 부팅·microSD 부하에 취약 → 세션 전 사전 점검 스크립트 돌리기
- 실패 팀 대비: 사전 빌드한 `.engine` 파일 공유 (빌드 실패가 실습 전체를 막지 않게)
- 팀 1개는 **의도적으로 INT8 mAP 손실을 크게** 만들어 QAT 필요성을 목격시키는 시나리오 배정
