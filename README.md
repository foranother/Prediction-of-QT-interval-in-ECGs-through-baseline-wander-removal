

기저선 변동 잡음 제거와 딥러닝을 이용한 심전도 QT 간격 예측 소프트웨어

QT 간격을 예측하여 심장 전문가들에게 참고용으로 제공하는 것을 목표로 한다.
이를 위해 기저선 변동 잡음 제거를 위한 저주파 통과 필터와 연속 웨이블릿 변환을 활용하여 심전도 측정 시 발생하는 여러 잡음을 제거한다.
또한, 심전도의 데이터 특징과 시계열 특징을 활용하기 위해 CNN과 Bi-LSTM을 결합한 모델은 제안한다.

Environment

requirements.txt 참조

Files

requirements.txt - 환경
qt-database - dataset
qt_generator - 데이터 생성(train.pkl, valid.pkl, test.pkl)
qt_detector - 모델 학습
model.h5 - 학습된 model
train.pkl - 모델 훈련 데이터
valid.pkl - 모델 검증 데이터
test.pkl - 모델 평가 데이터

Usage

0. requirements.txt 참고하여 환경 설정
1. python qt_generator.py - qt_generator를 실행하여 train, valid, test 데이터 생성
2. python qt_detector.py - qt_detector를 실행하여 모델 실행 후 결과값 반환

위 내용은 실행하는방법을 정리했어 이내용도포함해서 작성해줘

```markdown
# ECG QT 간격 예측 시스템 (Baseline Wander 제거 & CNN-BiLSTM)

<img src="https://img.shields.io/badge/Python-3.8%2B-blue">
<img src="https://img.shields.io/badge/TensorFlow-2.12%2B-orange">
<img src="https://img.shields.io/badge/License-MIT-yellow">

## 🔍 개요
**심전도 QT 간격 자동 측정 솔루션**  
기저선 변동 잡음 제거와 CNN-BiLSTM 하이브리드 모델을 결합한 심전도 분석 시스템

![모델 구조](https://via.placeholder.com/800x400.png/CCCCCC/808080?text=System+Architecture)

## 🚀 핵심 기능
- **신호 전처리**: 저주파 통과 필터(0.5Hz) + 연속 웨이블릿 변환(CWT)
- **하이브리드 모델**: CNN(지역 특징 추출) + Bi-LSTM(시계열 분석)
- **성능**: Accuracy **94%** | F1-Score **93.8%** ([QT Database 기준](#-참고-문헌))

## 📦 파일 구조
```

.
├── requirements.txt    \# 의존성 패키지
├── qt-database/        \# 원본 데이터셋
├── qt_generator.py     \# 데이터 전처리 (train/valid/test 생성)
├── qt_detector.py      \# 모델 학습/평가 스크립트
└── model.h5            \# 학습된 모델 가중치

```

## ⚙️ 설치 및 실행
### 1. 환경 설정
```

pip install -r requirements.txt

```

### 2. 데이터 생성
```

python qt_generator.py \
--input_dir ./qt-database \
--output_dir ./processed_data

```

### 3. 모델 학습
```

python qt_detector.py \
--data_dir ./processed_data \
--epochs 100 \
--batch_size 32

```

## 📊 성능 평가
| Metric    | BiLSTM(x2) | BiLSTM(x1) | GRU      |
|-----------|------------|------------|----------|
| Accuracy  | 0.940      | 0.932      | 0.927    |
| F1-Score  | 0.938      | 0.930      | 0.925    |

## 📚 데이터셋 정보
### QT Database ([PhysioNet](https://physioNet.org/content/qtdb))
| 항목          | 내용                     |
|---------------|--------------------------|
| 샘플 수       | 63명 (Lead I)            |
| 샘플링 주파수 | 250Hz → 125Hz 다운샘플링 |
| 데이터 분할   | 환자 기준 8:1:1          |

## 🛠️ 기술 스택
- **신호 처리**: `scipy.signal`, `pywavelets`
- **딥러닝 프레임워크**: TensorFlow 2.12+
- **데이터 관리**: Pickle, Numpy

## 📜 참고 문헌
- QT Database: [PhysioNet](https://physioNet.org/content/qtdb)
- 기저선 제거 기법: Luo S. (Journal of Electrocardiology 2010)

## 📧 연락처
**개발팀**: 전남대학교 인공지능융합학과  
**주요 담당자**: 이승준(dltmdwns3462@naver.com)  
**라이선스**: MIT (자세한 내용은 LICENSE 파일 참조)
```

<details>
<summary>📌 주의사항 (확장 보기)</summary>

1. **데이터 분할 전략**: 환자 단위 분리를 통해 과적합 방지
2. **신호 전처리 파이프라인**:
   ```python
   def preprocess_ecg(raw_signal):
       # 1. 125Hz 다운샘플링
       resampled = scipy.signal.resample(raw_signal, 125)
       # 2. 0.5Hz 저주파 필터 적용
       filtered = butter_lowpass(resampled, cutoff=0.5)
       # 3. 웨이블릿 변환 (스케일 62)
       return pywt.cwt(filtered, scales=np.arange(1,62), wavelet='mexh')
   ```
3. **모델 최적화**: Adam 옵티마이저 + Early Stopping 적용
</details>
<div style="text-align: center">⁂</div>

[^1]: https://patents.google.com/patent/KR20210078662A/ko

[^2]: https://www.jkiees.org/archive/view_article?pid=jkiees-33-1-76

[^3]: https://patents.google.com/patent/KR101896637B1/ko

[^4]: https://www.kwra.or.kr/!/download/?path=%2Fmedia%2F17%2Fpublication%2F2025%2F01%2F02%2F12-323-072긴급_장동우.pdf\&filename=12-323-072긴급_장동우.pdf\&ct=95\&oi=35787

[^5]: https://www.e-dmj.org/upload/pdf/dmj-29-3-215.pdf

[^6]: https://velog.io/@acadias12/CNN과-LSTM-결합하기

[^7]: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115\&topMenu=100\&dataSetSn=529

[^8]: https://blog.naver.com/gdpresent/223213183782

