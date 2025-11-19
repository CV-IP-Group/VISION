# Train sklearn intent model
# Written by ChatGPT and modified by CloudLake

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path


def _fit_classifier(data_dir: Path, model_dir: Path) -> None:
    train = pd.read_csv(data_dir / 'train.csv')
    dev = pd.read_csv(data_dir / 'dev.csv')
    test = pd.read_csv(data_dir / 'test.csv')

    X_train = pd.concat([train['text'], dev['text']], axis=0)
    y_train = pd.concat([train['label'], dev['label']], axis=0)

    vec = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 5),
        min_df=2,
        max_features=120000,
        sublinear_tf=True,
    )
    Xv = vec.fit_transform(X_train)

    base = LogisticRegression(max_iter=2000, C=2.0, class_weight='balanced', n_jobs=-1)
    try:
        clf = CalibratedClassifierCV(estimator=base, method='isotonic', cv=5)
    except TypeError:
        clf = CalibratedClassifierCV(base_estimator=base, method='isotonic', cv=5)

    clf.fit(Xv, y_train)

    Xte = vec.transform(test['text'])
    yte = test['label'].values
    pred = clf.predict(Xte)
    print(classification_report(yte, pred, digits=4))
    print('Confusion:\n', confusion_matrix(yte, pred))

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({'vec': vec, 'clf': clf, 'classes': clf.classes_}, model_dir / 'intent_sklearn.joblib')
    with open(model_dir / 'intent_threshold.json', 'w', encoding='utf-8') as f:
        json.dump({'tau_on': 0.75}, f, ensure_ascii=False, indent=2)
    print('Saved to', model_dir)


def main(data_dir: str = 'data', model_dir: str = 'models') -> None:
    data_path = Path(data_dir)
    model_path = Path(model_dir)
    _fit_classifier(data_path, model_path)


if __name__ == '__main__':
    main()
