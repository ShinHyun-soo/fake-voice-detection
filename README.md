# SW중심대학 경진 대회 / 가짜 음성 검출 및 탐지 (2024-07-01 ~ 2024-07-19)
5초 분량의 입력 오디오 샘플에서 영어 음성의 진짜(Real) 사람의 목소리와 생성 AI의 가짜(Fake) 사람의 목소리를 동시에 검출해내는 AI 모델 개발

### 1. 데이터 가공
- 학습 데이터 (단일 음성 데이터)
- 테스트 데이터 (합성 음성 데이터) - Unseen data, 학습에 사용할 수 없음.
- 데이터 재구축
- ![image](https://github.com/user-attachments/assets/ca82be1b-80d5-4b04-9cf3-7806b6cd5cb6)
- label -> [fake, real]
- 1 -> [0, 1] : 1명의 진짜 목소리만 존재 / 순수 학습 셋 에서 추출
- 2 -> [1, 0] : 1명의 가짜 목소리만 존재 / 순수 학습 셋 에서 추출
- 3 -> [1, 1] : 1명의 진짜 목소리와 1명의 가짜 목소리가 존재 / 0과 1을 랜덤으로 뽑아 긴 음성 길이에 맞게 짧은 음성에 제로 패딩을 삽입한 후 합성함.
- 4 -> [0, 1] :  2명의 진짜 목소리가 존재  / 1을 랜덤으로 뽑고 3과 동일.
- 5 -> [1, 0] : 2명의 가짜 목소리가 존재 / 0을 랜덤으로 뽑고 3과 동일.
- 6 -> [0, 0] : 아예 목소리가 없는 경우 / (사인파 등을 이용하여 랜덤 오디오 생성)


### 2. 모델 학습
- Hubert 를 avspoof 데이터 셋으로 미세 조정한 사전 학습 모델을 데이콘 데이터셋에 맞게 재학습 하였음.
- Layer-wise Learning Rate Decay, 레이어에 따라 학습률과 가중치 감쇠를 조절함.
- Layer Re-initialization, 뒤쪽부터 classifier, projector, top-1 layers 초기화.
- Longer Training and Frequent Validation, epoch 3, val_interval = 0.1
- Stratified cross validation 사용, n_fold = 20
- wav2vec 은 성능이 별로 안좋았음.

### 3. 진행 내용

| 주차  | 내용                              |
|-------|-----------------------------------|
| 0 ~ 1.5 | 모델 학습 및 추론 파이프라인을 구축하였음.|
| 1.5 ~ 2 | 모델이 자꾸 종속 사건 처럼 예측하여 이를 고치려 애를 먹음.|
| 2 ~ 2.5 | [1,1], [0,0] 같은 합성 데이터셋을 추가함으로써 위의 문제를 해결함. |
| 2.5 ~ 3 | 어쩌다 보니 실험 결과가 좋았는데, 왜 좋게 나왔는지 해석하고자 계속 실험을 하였음.  이 과정에서 모델 학습 및 추론 시간을 각종 트릭으로 극단적으로 줄이는데 성공함.|
| 3 ~  | 수많은 실험 결과 데이터셋의 수가 더 많았던 것이 성능을 향상시킨 요소로 생각하여, 데이터셋을 테스트 셋의 분포와 비슷하도록 구축하기 위해 노력하였음.|


### 4. 회고
오디오를 어떻게 처리할 것인지에 대해 고민을 많이 하지 않은 것이 패착으로 생각됨.
