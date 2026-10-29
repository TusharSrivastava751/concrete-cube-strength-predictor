"""Reproducible early-strength regression on synthetic cube results."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit


def positive(value, label):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(label + ' must be finite and positive')
    return number


def load_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required = {'sample_id', 'batch_id', 'strength_7d_mpa', 'strength_28d_mpa'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Missing required columns: ' + ', '.join(sorted(required)))
        rows = list(reader)
    seen = set()
    for row in rows:
        if not row['sample_id'].strip() or row['sample_id'] in seen or not row['batch_id'].strip():
            raise ValueError('Blank/duplicate sample ID or blank batch ID')
        seen.add(row['sample_id'])
        for key in ('strength_7d_mpa', 'strength_28d_mpa'):
            row[key] = positive(row[key], key)
    if len({r['batch_id'] for r in rows}) < 8:
        raise ValueError('At least eight distinct batches required for this demo split')
    return rows


def fit_evaluate(rows):
    x = np.array([[r['strength_7d_mpa']] for r in rows])
    y = np.array([r['strength_28d_mpa'] for r in rows])
    groups = [r['batch_id'] for r in rows]
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(x, y, groups))
    if np.ptp(x[train]) == 0:
        raise ValueError('Training feature has no variation')
    model = LinearRegression().fit(x[train], y[train])
    pred = model.predict(x[test])
    baseline = np.full(len(test), float(y[train].mean()))
    metrics = {'mae_mpa': float(mean_absolute_error(y[test], pred)),
               'rmse_mpa': float(np.sqrt(mean_squared_error(y[test], pred))),
               'training_mean_baseline_mae_mpa': float(mean_absolute_error(y[test], baseline))}
    result = {'scope': 'Synthetic demonstration only; no real-world performance claim',
              'sklearn_version': sklearn.__version__, 'split_seed': 42,
              'feature': 'strength_7d_mpa', 'target': 'strength_28d_mpa',
              'train_batches': sorted(set(groups[i] for i in train)),
              'test_batches': sorted(set(groups[i] for i in test)),
              'train_samples': [rows[i]['sample_id'] for i in train],
              'test_samples': [rows[i]['sample_id'] for i in test],
              'coefficient': float(model.coef_[0]), 'intercept': float(model.intercept_),
              'training_range_7d_mpa': [float(x[train].min()), float(x[train].max())],
              'metrics': metrics,
              'held_out_predictions': [{'sample_id': rows[i]['sample_id'], 'actual_28d_mpa': float(y[i]),
                                        'predicted_28d_mpa': float(p)} for i, p in zip(test, pred)]}
    return result


def screen(model, strength_7d, target_28d):
    strength = positive(strength_7d, '7-day strength')
    target = positive(target_28d, 'screening target')
    low, high = model['training_range_7d_mpa']
    if not low <= strength <= high:
        return {'status': 'abstain_outside_training_range', 'prediction_mpa': None,
                'note': 'Obtain engineering review; the demo does not extrapolate'}
    prediction = model['coefficient'] * strength + model['intercept']
    return {'status': 'below_screening_target' if prediction < target else 'at_or_above_screening_target',
            'prediction_mpa': prediction, 'screening_target_mpa': target,
            'note': 'No acceptance decision or calibrated failure probability; verify actual 28-day tests'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data', type=Path)
    p.add_argument('--early', type=float, default=24)
    p.add_argument('--target', type=float, default=40)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    try:
        result = fit_evaluate(load_rows(args.data))
        result['input_sha256'] = hashlib.sha256(args.data.read_bytes()).hexdigest()
        result['screening_example'] = screen(result, args.early, args.target)
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
        metrics = result['metrics']
        (args.out / 'report.md').write_text('# Synthetic cube-strength model evaluation\n\n'
            '**Synthetic data only. These metrics do not establish site performance.**\n\n'
            f"Training batches: {len(result['train_batches'])}; held-out batches: {len(result['test_batches'])}.\n\n"
            f"MAE: {metrics['mae_mpa']:.3f} MPa; RMSE: {metrics['rmse_mpa']:.3f} MPa.\n\n"
            f"Training-mean baseline MAE: {metrics['training_mean_baseline_mae_mpa']:.3f} MPa.\n\n"
            f"Screening example: `{result['screening_example']['status']}`.\n\n"
            'See evaluation.json for batch membership, every prediction, parameters and source hash.\n')
        print(json.dumps({'metrics': metrics, 'screening_example': result['screening_example']}, indent=2))
    except (ValueError, OSError) as exc:
        p.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
