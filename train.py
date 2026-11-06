#!/usr/bin/env python3
"""Train an output-token predictor using only input_tokens."""
import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from inspect_data import load, summarize


def usable(row):
    if row.get('status') != 'ok':
        return False
    for k in ('input_tokens', 'output_tokens'):
        v = row.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            return False
    return True


def metrics(y, pred):
    pred = np.maximum(0, pred)
    return {'mae': float(mean_absolute_error(y, pred)),
            'rmse': float(np.sqrt(mean_squared_error(y, pred))),
            'r2': float(r2_score(y, pred)),
            'underprediction_rate': float(np.mean(pred < y)),
            'mean_error': float(np.mean(pred - y))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='artifacts')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rows = load(args.data)
    data = [r for r in rows if usable(r)]
    if len(data) < 50:
        raise SystemExit('Need at least 50 successful rows with valid token counts.')
    ids = [r.get('id') for r in data if r.get('id') is not None]
    if len(ids) != len(set(ids)):
        raise SystemExit('Duplicate IDs detected; resolve them before training to prevent leakage.')
    fingerprints = {(r.get('model_digest') or r.get('model'), json.dumps(r.get('settings'), sort_keys=True)) for r in data}
    if len(fingerprints) != 1:
        raise SystemExit('Multiple model versions or generation settings detected; filter to one configuration before training.')
    x = np.array([[r['input_tokens']] for r in data], dtype=float)
    y = np.array([r['output_tokens'] for r in data], dtype=float)
    indices = np.arange(len(data))
    trainval, test = train_test_split(indices, test_size=0.2, random_state=args.seed)
    train, val = train_test_split(trainval, test_size=0.125, random_state=args.seed)
    candidates = {
        'median': DummyRegressor(strategy='median'),
        'mean': DummyRegressor(strategy='mean'),
        'linear': LinearRegression(),
        'boosted_median': HistGradientBoostingRegressor(loss='absolute_error', max_iter=150, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1, random_state=args.seed),
        'boosted_mean': HistGradientBoostingRegressor(loss='squared_error', max_iter=150, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1, random_state=args.seed),
    }
    validation = {}
    for name, model in candidates.items():
        model.fit(x[train], y[train])
        validation[name] = metrics(y[val], model.predict(x[val]))
    selected = min(validation, key=lambda n: validation[n]['mae'])
    test_metrics = {}
    for name, model in candidates.items():
        model.fit(x[trainval], y[trainval])
        test_metrics[name] = metrics(y[test], model.predict(x[test]))
    settings = data[0].get('settings') or {}
    metadata = {'model_name': data[0].get('model'), 'model_digest': data[0].get('model_digest'),
                'settings': settings, 'input_min': float(x.min()), 'input_max': float(x.max()),
                'target': 'observed output_tokens, including generation length limits', 'sklearn_version': sklearn.__version__}
    report = {'audit': summarize(rows), 'excluded_rows': len(rows) - len(data),
              'split': {'train': len(train), 'validation': len(val), 'test': len(test), 'seed': args.seed,
                        'method': 'random row split; related prompts can cross splits'},
              'selected_by_validation_mae': selected, 'validation': validation, 'test': test_metrics,
              'metadata': metadata}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({'estimator': candidates[selected], 'metadata': metadata}, out / 'model.joblib')
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'selected': selected, 'test': test_metrics, 'saved_to': str(out)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
