# Qwen3 출력 토큰 예측

Ollama 수집 결과의 `input_tokens`만으로 해당 설정에서 관측되는 `output_tokens`를 예측합니다. 답변 내용이나 생성 이후의 정보는 입력에 사용하지 않습니다. `status=ok`이면서 토큰 수가 유효한 행을 사용합니다. 중복 ID나 서로 다른 모델 버전·생성 설정이 섞이면 학습을 중단합니다. `done_reason=length`인 행도 실제 제한된 출력값으로 포함합니다.

## 서버에서 실행

현재 서버에 `~/qwen3_8b_100k_questions_bundle/token_predictor`와 학습 결과를 설치했습니다. 결과를 다시 만들려면:

```bash
cd ~/qwen3_8b_100k_questions_bundle/token_predictor
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python inspect_data.py ../results_10000.jsonl
OMP_NUM_THREADS=2 .venv/bin/python train.py --data ../results_10000.jsonl --output artifacts
.venv/bin/python predict.py --model artifacts/model.joblib 100 500 1000
```

첫 실행 시 가상환경 생성과 설치가 필요합니다. 이미 생성된 서버 가상환경을 쓰려면 `.venv/bin/python` 대신 `~/qwen3_8b_100k_questions_bundle/.token_predict_venv/bin/python`을 사용해도 됩니다. 입력은 Ollama에서 측정한 토큰 수와 같은 기준이어야 합니다. 질문의 글자 수가 아닙니다.

예측은 입력 하나당 JSON 한 줄을 출력합니다. `outside_training_range`는 학습 데이터의 전체 입력 토큰 범위를 벗어났다는 경고입니다. 학습 결과는 `artifacts/model.joblib`과 `artifacts/report.json`에 저장됩니다. 다른 데이터는 `--data`로 지정할 수 있습니다. 신뢰할 수 있는 모델 파일만 불러오세요. joblib 파일을 만들고 사용할 때 같은 버전의 Python과 라이브러리를 사용하는 것이 좋습니다.

## 평가와 해석

고정 시드로 70% 학습, 10% 검증, 20% 테스트로 무작위 분할합니다. 중앙값·평균·선형 회귀·두 가지 부스팅 모델을 검증 MAE로 선택한 뒤 학습+검증 데이터로 다시 학습해 테스트합니다. 보고서에 모든 후보의 MAE, RMSE, R², 과소예측 비율을 기록합니다. 테스트 결과를 후보 선택에 사용하지 않습니다.

현재 10,000건에서 입력 범위는 38~3,797토큰이고 23건은 4,096토큰 제한에 도달했습니다. 모델과 설정은 Qwen3 8B Q4_K_M, temperature 0.7, num_predict 4096 등 수집 파일에 기록된 값에 한정됩니다. 예측값은 평균적인 추정치이며 출력 상한이나 보장값이 아닙니다. 한도에 도달한 결과는 실제로 더 길었을 수 있으므로 제한 없는 출력 길이를 예측하는 용도로 해석하면 안 됩니다.

실제 학습 결과(시드 42, 테스트 2,000건): 검증 MAE로 선택된 `boosted_mean`의 테스트 MAE는 555.2토큰, RMSE는 667.6토큰, R²는 0.069, 과소예측 비율은 52.5%입니다. 학습 데이터 중앙값만 예측하는 기준 MAE는 573.6토큰이므로 개선 폭은 약 3.2%입니다. 입력 길이 하나만으로는 출력 길이를 정밀하게 설명하기 어렵습니다.

무작위 분할에서는 비슷한 템플릿의 질문이 학습과 테스트 양쪽에 들어갈 수 있어 새로운 유형의 질문에서 성능이 더 낮을 수 있습니다. 결과 파일에는 템플릿 그룹 정보가 없어 그룹 분할은 적용하지 않았습니다. 다른 모델, 프롬프트 구성, 생성 설정, 실제 운영 질문에 적용하기 전 별도 데이터로 검증하세요.
