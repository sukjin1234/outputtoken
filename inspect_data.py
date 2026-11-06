#!/usr/bin/env python3
"""Inspect collected JSONL without exposing prompts or answers."""
import argparse
import collections
import json
import statistics


def load(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'Invalid JSON at line {line_no}: {e}') from e
            if not isinstance(row, dict):
                raise ValueError(f'Line {line_no} is not an object')
            rows.append(row)
    return rows


def summarize(rows):
    ids = [r.get('id') for r in rows if r.get('id') is not None]
    result = {'rows': len(rows), 'duplicate_ids': len(ids) - len(set(ids))}
    for key in ('status', 'done_reason', 'model', 'model_digest'):
        result[key] = dict(collections.Counter(str(r.get(key)) for r in rows))
    result['settings'] = dict(collections.Counter(json.dumps(r.get('settings'), sort_keys=True, ensure_ascii=False) for r in rows))
    for key in ('input_tokens', 'output_tokens'):
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
        result[key] = {'valid': len(values), 'min': min(values) if values else None,
                       'median': statistics.median(values) if values else None,
                       'max': max(values) if values else None}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    args = parser.parse_args()
    print(json.dumps(summarize(load(args.data)), ensure_ascii=False, indent=2))
