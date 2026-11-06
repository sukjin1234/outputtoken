#!/usr/bin/env python3
import argparse
import json
import math
import joblib

parser = argparse.ArgumentParser(description='Predict output tokens from input token counts.')
parser.add_argument('--model', default='artifacts/model.joblib')
parser.add_argument('input_tokens', nargs='+', type=int)
args = parser.parse_args()
if any(n < 0 for n in args.input_tokens):
    parser.error('input token counts must be nonnegative')
bundle = joblib.load(args.model)
meta = bundle['metadata']
predictions = bundle['estimator'].predict([[n] for n in args.input_tokens])
for n, pred in zip(args.input_tokens, predictions):
    print(json.dumps({'input_tokens': n, 'predicted_output_tokens': max(0, round(float(pred))),
                      'outside_training_range': n < meta['input_min'] or n > meta['input_max']}, ensure_ascii=False))
