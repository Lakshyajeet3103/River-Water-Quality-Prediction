"""Train a small oxygen-balance PINN on training-only initial conditions."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/oxygen_pinn_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from three_model_support import load, sha
from river_water_quality.training.oxygen_pinn import fit, predict, initial_demand, analytical, KD, KR, CS


def main():
    train, seen, strict, examples, models, hashes = load()
    out = ROOT / 'docs/oxygen_pinn_results'
    out.mkdir(parents=True, exist_ok=True)
    print('Training oxygen-balance PINN on 80 training initial conditions...', flush=True)
    checkpoint, history = fit(train.ground_bod, train.ground_dissolved_oxygen)
    path = out / 'oxygen_pinn.pt'
    torch.save(checkpoint, path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    train[['observation_id', 'ground_bod', 'ground_dissolved_oxygen']].to_csv(out / 'training_initial_conditions.csv', index=False)
    pd.DataFrame(history).to_csv(out / 'training_history.csv', index=False)
    rows = []
    # Test 46 conditions per saved model against the analytical equation solution.
    for cohort, frame in [('familiar_20', seen), ('unseen_strict_26', strict)]:
        for name, model in models.items():
            bods = model['predict'](frame[model['features']])
            for (_, row), bod in zip(frame.iterrows(), bods):
                times = np.linspace(0, 5, 21)
                state = predict(checkpoint, times, bod, row.ground_dissolved_oxygen)
                exact_l, exact_o = analytical(times, initial_demand(bod), row.ground_dissolved_oxygen)
                for i, t in enumerate(times):
                    rows.append({'model': name, 'cohort': cohort, 'observation_id': row.observation_id,
                        'station_id': row.station_id, 'time_days': t, 'predicted_bod5_scenario_mg_l': float(bod),
                        'initial_measured_do_mg_l': float(row.ground_dissolved_oxygen),
                        'pinn_remaining_demand_mg_l': float(state[i, 0]), 'pinn_do_mg_l': float(state[i, 1]),
                        'analytical_remaining_demand_mg_l': float(exact_l[i]), 'analytical_do_mg_l': float(exact_o[i])})
            if sha(model['path']) != model['sha256']:
                raise ValueError('Saved BOD model changed')
    data = pd.DataFrame(rows)
    data['absolute_oxygen_equation_error_mg_l'] = (data.pinn_do_mg_l-data.analytical_do_mg_l).abs()
    data.to_csv(out / 'oxygen_scenarios.csv', index=False)
    metrics = []
    for (name, cohort), group in data.groupby(['model', 'cohort']):
        metrics.append({'model': name, 'cohort': cohort, 'initial_conditions': group.observation_id.nunique(),
            'time_points_per_condition': 21, 'oxygen_solver_mae_mg_l': float(group.absolute_oxygen_equation_error_mg_l.mean()),
            'maximum_oxygen_solver_error_mg_l': float(group.absolute_oxygen_equation_error_mg_l.max()),
            'negative_predicted_oxygen_points': int(group.pinn_do_mg_l.lt(0).sum())})
    pd.DataFrame(metrics).to_csv(out / 'solver_metrics.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].semilogy([r['epoch'] for r in history], [r['ode_loss'] for r in history])
    axes[0].set(xlabel='Training epoch', ylabel='ODE residual loss', title='PINN learns oxygen-balance equations')
    first = examples.observation_id.iloc[0]
    for name, group in data[(data.observation_id == first) & (data.cohort == 'familiar_20')].groupby('model'):
        line, = axes[1].plot(group.time_days, group.pinn_do_mg_l, label=name)
        axes[1].plot(group.time_days, group.analytical_do_mg_l, '--', color=line.get_color())
    axes[1].set(xlabel='Assumed elapsed days', ylabel='Scenario oxygen (mg/L)', title='Solid: PINN; dashed: analytical reference')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'oxygen_pinn_demo.png', dpi=140)
    plt.close(fig)
    report = {'scope': 'Optional physics-informed oxygen simulator, not BOD-model retraining',
        'equations': ['dL/dt = -kd*L', 'dO/dt = kr*(Cs-O)-kd*L'],
        'assumptions': {'kd_per_day': KD, 'kr_per_day': KR, 'Cs_mg_l': CS,
            'BOD_conversion': 'Treat supplied/predicted BOD as BOD5 only for this scenario; L0=BOD5/(1-exp(-kd*5))'},
        'training_rows': 80, 'epochs': checkpoint['epochs'], 'initial_ode_loss': history[0]['ode_loss'],
        'final_logged_ode_loss': history[-1]['ode_loss'], 'model_sha256': sha(path),
        'frozen_bod_models': {n: m['sha256'] for n, m in models.items()}, 'source_sha256': hashes,
        'limitations': ['Oxygen and remaining demand are simulated, not observed future water quality.',
            'No measured oxygen trajectories, calibrated rates, travel times or flow measurements are available.',
            'Analytical comparison validates the equation solver, not river forecast accuracy.',
            'Assumed rates and saturation ignore site-specific temperature, pH, photosynthesis, other loads and mixing.',
            'BOD laboratory incubation convention is not verified; BOD5 conversion is a scenario assumption.',
            'Satellite-only BOD prediction remains satellite-only; the optional oxygen simulator additionally requires measured DO.',
            'All three BOD checkpoints and their prediction accuracy remain unchanged.']}
    report['output_sha256'] = {p.name: sha(p) for p in out.iterdir() if p.name != 'provenance.json'}
    (out / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(pd.DataFrame(metrics).to_string(index=False))
    print('ODE loss:', history[0]['ode_loss'], '->', history[-1]['ode_loss'])


if __name__ == '__main__':
    main()
