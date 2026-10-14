"""Train one switchable network or five fixed demonstration variants on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('outputs/matplotlib_cache').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from river_water_quality.training.experimental_network import Switches, fit_network, predict_network, select_epochs
from river_water_quality.training.sklearn_demonstrations import ALL, check_separation
from river_water_quality.training.sklearn_fusion import TARGET, regression_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--constraint', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--station-adversarial', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--suite', action='store_true', help='Baseline, each switch alone, and all switches together')
    parser.add_argument('--max-epochs', type=int, default=400)
    parser.add_argument('--output', default='outputs/experimental_network_20261014')
    args = parser.parse_args()
    if not 1 <= args.max_epochs <= 1000:
        raise ValueError('max-epochs must be 1–1000')
    out = Path(args.output)
    if out.exists():
        raise ValueError('Output exists; choose a new directory')
    data_path = Path('outputs/india_100_chemical_dataset_20261014/paired.csv')
    data = pd.read_csv(data_path)
    provenance = json.loads(data_path.with_name('provenance.json').read_text())
    if sha(data_path) != provenance['output_sha256']:
        raise ValueError('Dataset hash mismatch')
    split_path = Path('outputs/satellite_only_80_20_20261014/split_assignments.csv')
    splits = pd.read_csv(split_path)[['observation_id', 'split']]
    if set(data.observation_id) != set(splits.observation_id) or not splits.observation_id.is_unique:
        raise ValueError('Saved splits do not cover exactly the input events')
    data = data.merge(splits, on='observation_id', validate='one_to_one').sort_values(['sample_date', 'station_id', 'observation_id'])
    train, test = data.query("split == 'train'"), data.query("split == 'test'")
    if len(train) != 80 or len(test) != 20:
        raise ValueError('Expected the saved 80/20 split')
    check_separation(train, test)
    preserved = [Path('outputs/satellite_only_80_20_20261014/model.joblib'),
                 Path('outputs/fusion_sklearn_expanded_20261014/fusion_model.joblib')]
    preserved_hashes = {str(p): sha(p) for p in preserved}
    variants = {'baseline': Switches(), 'gate_only': Switches(gate=True),
                'constraint_only': Switches(constraint=True),
                'station_adversarial_only': Switches(station_adversarial=True),
                'all_components': Switches(True, True, True)} if args.suite else {
                    'custom': Switches(args.gate, args.constraint, args.station_adversarial)}
    out.mkdir(parents=True)
    metrics, reports, sweeps, histories = [], {}, [], {}
    for name, switches in variants.items():
        print(f'{name}: selecting training duration within training data...', flush=True)
        epochs, validation_mae, selection_history, inner = select_epochs(train, switches, args.max_epochs)
        checkpoint, history = fit_network(train, switches, epochs, args.max_epochs)
        checkpoint['input_sha256'] = sha(data_path)
        checkpoint['split_sha256'] = sha(split_path)
        checkpoint['selection_max_epochs'] = args.max_epochs
        checkpoint['validation_mae'] = validation_mae
        model_path = out / f'{name}.pt'
        torch.save(checkpoint, model_path)
        restored = torch.load(model_path, map_location='cpu', weights_only=True)
        results = predict_network(restored, test)
        np.testing.assert_array_equal(results['predicted'], predict_network(checkpoint, test)['predicted'])
        # No BOD label or station ID is required by the prediction path.
        np.testing.assert_array_equal(results['predicted'], predict_network(restored, test[[*ALL, 'temporal_offset_days']])['predicted'])
        row = {'model': name, 'epochs': epochs, 'validation_mae': validation_mae,
               **regression_metrics(test[TARGET], results['predicted'])}
        metrics.append(row)
        predictions = test[['observation_id', 'station_id', 'station_name', 'sample_date', 'temporal_offset_days']].copy()
        predictions['actual_bod_mg_l'] = test[TARGET]
        predictions['raw_bod_mg_l'] = results['raw']
        predictions['predicted_bod_mg_l'] = results['predicted']
        predictions['absolute_error_mg_l'] = np.abs(test[TARGET].to_numpy()-results['predicted'])
        predictions['satellite_weight'] = results['satellite_weight']
        predictions['ground_weight'] = results['ground_weight']
        if results['station_class'] is not None:
            predictions['classifier_station_prediction'] = [checkpoint['station_classes'][i] for i in results['station_class']]
        predictions.to_csv(out / f'{name}_predictions.csv', index=False)
        history.to_csv(out / f'{name}_training_history.csv', index=False)
        selection_history.to_csv(out / f'{name}_selection_history.csv', index=False)
        inner.to_csv(out / f'{name}_selection_splits.csv', index=False)
        histories[name] = history
        sweep = pd.concat([test.iloc[:1].copy()]*8, ignore_index=True)
        sweep['temporal_offset_days'] = np.arange(8)
        changing = predict_network(restored, sweep)
        for gap in range(8):
            sweeps.append({'model': name, 'absolute_gap_days': gap,
                           'satellite_weight': float(changing['satellite_weight'][gap]),
                           'ground_weight': float(changing['ground_weight'][gap]),
                           'predicted_bod_mg_l': float(changing['predicted'][gap]),
                           'example_observation_id': test.observation_id.iloc[0]})
        reports[name] = {'switches': checkpoint['switches'], 'test_metrics': row,
                         'model_sha256': sha(model_path), 'weight_changes': checkpoint['weight_changes'],
                         'epochs_with_constraint_penalty': int(history.constraint_loss_applied.gt(0).sum()),
                         'maximum_negative_bod_penalty': float(history.negative_bod_penalty.max()),
                         'test_negative_raw_predictions': int((results['raw'] < 0).sum()),
                         'station_classifier_training_accuracy': float(history.station_accuracy.iloc[-1]) if switches.station_adversarial else None,
                         'constraint_weight': 1.0, 'station_loss_weight': .1, 'maximum_reversal_strength': .2}
        if switches.gate and not any(v > 0 for k, v in checkpoint['weight_changes'].items() if k.startswith('gate_layer')):
            raise ValueError('Gate parameters did not learn')
        if switches.station_adversarial and not any(v > 0 for k, v in checkpoint['weight_changes'].items() if k.startswith('station_head')):
            raise ValueError('Station classifier did not learn')
        print(f"  epochs={epochs}, test MAE={row['mae']:.4f} mg/L", flush=True)
    baseline = np.full(len(test), train[TARGET].mean())
    metrics.append({'model': 'training_mean', 'epochs': 0, 'validation_mae': None,
                    **regression_metrics(test[TARGET], baseline)})
    pd.DataFrame(metrics).to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(sweeps).to_csv(out / 'gate_sweep.csv', index=False)
    splits.to_csv(out / 'split_assignments.csv', index=False)
    if {str(p): sha(p) for p in preserved} != preserved_hashes:
        raise ValueError('Existing models changed')
    report = {'scope': 'Small experimental two-branch network with switchable mechanisms',
              'train_rows': 80, 'test_rows': 20, 'inner_selection': 'Chronological 60/20 inside training; first valid separated boundary',
              'features': ALL, 'target': TARGET, 'input_sha256': sha(data_path), 'split_sha256': sha(split_path),
              'preserved_model_hashes': preserved_hashes, 'variants': reports, 'torch_version': str(torch.__version__),
              'parameter_count': 130, 'seed': 42, 'cpu_only': True,
              'limitations': ['All cases are from three known Ganga stations; station invariance is not foreign-river adaptation.',
                              'The constraint is nonnegative BOD, not a differential-equation PINN.',
                              'Date-dependent branch weights are learned representation weights, not calibrated confidence or SHAP explanations.',
                              'Station ID is used only as the auxiliary training label; no held-out features or BOD labels enter optimization.',
                              'Previously examined pilot test cases; no component is promised to improve accuracy.',
                              'Nonnegative training penalty can become inactive once predictions are positive; it is recorded rather than fabricated.',
                              'All model variants clip negative inference predictions; raw predictions are also retained.',
                              'Existing source-method, time/coordinate and water-pixel assumptions remain.']}
    (out / 'run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    draw(out, histories, pd.DataFrame(sweeps), pd.DataFrame(metrics))
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f'Saved experiments to {out}')


def draw(out, histories, sweeps, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        axes[0].plot(h.epoch, h.bod_mse, label=name)
        axes[1].plot(h.epoch, h.constraint_loss_applied, label=name)
    axes[0].set_title('BOD training loss'); axes[1].set_title('Applied nonnegative penalty')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / 'training_losses.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, rows in sweeps.groupby('model'):
        ax.plot(rows.absolute_gap_days, rows.satellite_weight, marker='o', label=name)
    ax.set(xlabel='Absolute satellite–sampling gap (days)', ylabel='Satellite branch weight', ylim=(0, 1))
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / 'gate_weights.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(metrics.model, metrics.mae); ax.set_xlabel('Test MAE (mg/L; lower is better)')
    fig.tight_layout(); fig.savefig(out / 'comparison.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, h in histories.items():
        if h.station_cross_entropy.gt(0).any():
            ax.plot(h.epoch, h.station_cross_entropy, label=name)
    ax.set(xlabel='Epoch', ylabel='Station classifier cross-entropy')
    if ax.lines:
        ax.legend()
    fig.tight_layout(); fig.savefig(out / 'station_adversarial_loss.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()
