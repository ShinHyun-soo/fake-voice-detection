# SW중심대학 디지털 경진대회_생성 AI의 가짜(Fake) 음성 검출 및 탐지
- 'facebook/hubert-base-ls960' 를 MultiLabelSoftMarginLoss 를 이용하여, Multi-Label Classification Task 에 맞게 Fine-tuning 하였음.

# 평가 지표
- 0.5 × (1 − AUC) + 0.25 × Brier + 0.25 × ECE
  - AUC(Area Under the Curve)
  - Brier Score
  - ECE(Expected Calibration Error)


# 실행 환경
- Windows 11
- Python 3.10
- Cuda 12.1
- PyTorch 2.3.1
- CPU : Ryzen 5 5600
- Memory Used : 48.3/58.0GB(32GB + Virtual Memory)
- RTX 4060TI 16GB(15.5GB 사용)
