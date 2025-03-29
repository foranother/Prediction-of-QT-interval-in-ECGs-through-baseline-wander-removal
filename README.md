# 기저선 변동 잡음 제거와 딥러닝을 이용한 심전도 QT 간격 예측

```markdown
# 기저선 변동 잡음 제거와 딥러닝을 이용한 심전도 QT 간격 예측

이 저장소는 기저선 변동 잡음 제거와 CNN-BiLSTM 하이브리드 딥러닝 모델을 활용하여 심전도 신호의 QT 간격을 예측하는 소프트웨어를 제공합니다.

## 📄 설명

본 프로젝트는 심전도 신호에서 QT 간격을 예측하여 심장 전문가들에게 참고 정보를 제공하는 것을 목표로 합니다.  
이를 위해 저주파 통과 필터와 연속 웨이블릿 변환(CWT)을 사용하여 기저선 변동 잡음을 제거하고, CNN-BiLSTM 모델을 통해 심전도의 공간적 및 시계열적 특징을 분석합니다.

## 주요 기능
- 잡음 제거: 저주파 통과 필터와 CWT를 사용하여 기저선 변동 잡음 제거
- 딥러닝 모델: CNN으로 공간적 특징 추출, Bi-LSTM으로 시계열 의존성 학습
- 높은 성능: QT Database 기준 Accuracy 94%, F1-Score 93.8%

## 환경 설정
필요한 패키지는 `requirements.txt` 파일을 참고하세요.

## 파일 구조
```

.
├── requirements.txt    \# 의존성 패키지 목록
├── qt-database/        \# 원본 QT 데이터셋
├── qt_generator.py     \# 데이터 전처리 스크립트 (train/valid/test 생성)
├── qt_detector.py      \# 모델 학습 및 평가 스크립트
├── model.h5            \# 학습된 모델 가중치 파일
├── train.pkl           \# 훈련 데이터셋
├── valid.pkl           \# 검증 데이터셋
└── test.pkl            \# 테스트 데이터셋

```

## 사용 방법

### 1. 환경 설정
필요한 패키지를 설치합니다:
```

pip install -r requirements.txt

```

### 2. 데이터 생성
다음 명령어를 실행하여 원본 QT 데이터를 전처리하고, train/valid/test 데이터를 생성합니다:
```

python qt_generator.py --input_dir ./qt-database --output_dir ./processed_data

```

### 3. 모델 학습
다음 명령어를 실행하여 CNN-BiLSTM 모델을 학습합니다:
```

python qt_detector.py --data_dir ./processed_data --epochs 100 --batch_size 32

```

### 4. 모델 평가
학습된 모델(`model.h5`)을 테스트 데이터로 평가합니다:
```

python qt_detector.py --evaluate --model_path ./model.h5 --test_data ./processed_data/test.pkl

```

## 성능

| 평가 지표   | 값      |
|-------------|---------|
| Accuracy    | 94.0%   |
| Precision   | 97.7%   |
| Recall      | 90.2%   |
| F1-Score    | 93.8%   |

## 데이터셋

본 프로젝트는 [PhysioNet](https://physionet.org/content/qtdb/1.0.0/)의 QT Database를 사용합니다:
- 샘플링 주파수: 원본 250Hz → 다운샘플링 후 125Hz 사용
- 데이터 분할: 환자 단위로 train(80%), validation(10%), test(10%) 구성

## 모델 구조

제안된 모델은 CNN과 Bi-LSTM을 결합하여 심전도 신호의 공간적 및 시계열적 특징을 효과적으로 학습합니다.

### 주요 구성 요소:
1. CNN 레이어: 신호의 지역적 특징 추출  
2. Bi-LSTM 레이어: 양방향 시계열 의존성 학습  
3. Dense 레이어: 이진 분류 수행 (QT 간격 여부)

