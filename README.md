# Fake Currency Detection Using Deep Learning (CNN)

## Project Structure
```
fake-currency-project/
  dataset/
    training/
      fake/
      real/
    validation/
      fake/
      real/
    testing/
      fake/
      real/
  src/
    train.py
    evaluate.py
    predict.py
  models/
  outputs/
  app.py
  requirements.txt
  README.md
```

## Setup
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```
python src/train.py
```

## Evaluate
```
python src/evaluate.py
```

## Predict One Image
```
python src/predict.py path/to/image.jpg
```

## Run UI (Streamlit)
```
streamlit run app.py
```

## Outputs
- Trained model: `models/fake_currency_mobilenetv2.keras`
- Class mapping: `outputs/class_names.json`
- Training history: `outputs/training_history.json`
- Evaluation metrics: `outputs/evaluation.json`
- Best threshold: `outputs/best_threshold.json`
<<<<<<< HEAD
.
=======
>>>>>>> fc2597c49205c316702e16b0d9d347c783f5fd37
