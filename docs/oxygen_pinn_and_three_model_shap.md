# Oxygen PINN and SHAP for the three presentation models

## What was added

1. Actual Kernel SHAP explanations for the selected four-input MLP, the frozen
   all-components two-branch network, and the selected satellite ratio MLP.
2. A small conditional neural network trained using oxygen-balance differential
   equations. It is an optional simulator after BOD prediction. The three saved
   BOD checkpoints, their weights and accuracy are unchanged.

These additions are in the isolated `unseen-model-evaluation` workspace/branch.
They are not yet integrated into the frontend or committed/pushed.

## SHAP

Each model explains the same seven examples: the first saved test event from
each of the three familiar stations and four new stations. The new cases exclude
satellite acquisitions shared with training. The background is a fixed seed-42
sample of 20 of the 80 training rows; no test rows enter the background.

| Model | Inputs explained | Outputs saved |
|---|---|---|
| Selected-feature combined | B8, B8A, B11 and fecal coliform | Seven local explanations, sample ranking and chart |
| All-components two-branch | Ten bands, fecal, pH, oxygen and date gap | Same; satellite/ground/date-gap contributions separate |
| Improved satellite-only | Ten raw bands, with ratios recalculated inside prediction | Same; zero ground-input contribution |

Kernel SHAP uses 512 requested samples and no feature-selection regularisation.
The four-feature case can enumerate all nontrivial feature coalitions. Other
cases are sampling approximations. Each explanation reconstructs its model's
prediction as training-background reference plus summed contributions, within
1e-5 mg/L. Frozen checkpoint hashes are checked before/after use.

The ratio model's SHAP does not independently perturb derived ratios; every
masked band combination produces its corresponding ratios. This preserves
the implemented feature relationships. These explanations refer to BOD, not
the custom category calculation or the oxygen PINN. Seven-case mean absolute
SHAP rankings are demonstration summaries, not population-wide importance or
new feature selection. Correlated bands and unrealistic masked combinations
limit causal interpretation. Gate weights and SHAP contributions are different.

Charts:

- [Selected-feature model](three_model_shap_results/selected_four_inputs_shap.png)
- [All-components network](three_model_shap_results/all_components_shap.png)
- [Satellite ratio model](three_model_shap_results/satellite_ratios_shap.png)

Source: [official KernelExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html).

## The simple oxygen PINN

State variables are remaining oxygen demand `L(t)` and dissolved oxygen `O(t)`,
both in mg/L; time is in days. The network has two 24-neuron tanh hidden layers.
It takes elapsed time, starting demand and starting measured oxygen and predicts
both state variables. Initial conditions are satisfied exactly by construction.

The differentiable physics losses enforce:

```
dL/dt = -kd * L
dO/dt = kr * (Cs - O) - kd * L
```

The first equation consumes remaining demand. The second balances oxygen
consumption against replenishment toward an assumed saturation. Automatic
differentiation calculates the derivatives; training minimizes squared equation
residuals plus a negative-state penalty. There is no training trajectory label.

Fixed illustrative parameters, not calibrated estimates:

- `kd = 0.23/day`, using the natural-exponential convention.
- `kr = 0.50/day`.
- `Cs = 8.26 mg/L`, an approximate freshwater saturation near 25 C at sea level.
- Five-day scenario horizon.
- BOD is treated as BOD5 **for this scenario only**, with
  `L0 = BOD5 / (1 - exp(-kd * 5))`. The source laboratory incubation convention
  has not been verified.

This follows simplified first-order BOD/oxygen-balance kinetics described in
the [EPA BOD/DO technical guidance](https://www.epa.gov/sites/default/files/2019-12/documents/technical-guidance-tmdl-book2.pdf).
The chosen rates are project assumptions, not values measured at these stations.

The PINN trains for 1,500 fixed epochs using only the 80 training events' BOD and
DO as initial-condition pairs, with 11 collocation times per pair. Those 880
equation-evaluation points are not 880 real observations. The original BOD model
test cases are excluded from PINN training. Test scenarios use each frozen BOD
model's prediction and the corresponding measured starting DO.

The oxygen equation has a closed-form analytical solution. Comparing against
that solution tests whether the PINN solves the stated equations; it does not
test real river forecast accuracy. On 20 familiar and 26 new-station initial
conditions per BOD model (21 time points each), mean oxygen-solver error ranges
from **0.007 to 0.011 mg/L**. Maximum error is about **0.059 mg/L**. No tested
scenario produced negative predicted oxygen. Logged ODE loss fell from 1.028 to
0.000318. These are solver checks, not improvements to BOD prediction accuracy.

[Open the PINN training/oxygen scenario chart](oxygen_pinn_results/oxygen_pinn_demo.png).

Missing real information includes future oxygen measurements, flow/travel time,
site-specific rates, water temperature and other oxygen sources/sinks. This
simple system omits mixing, photosynthesis, nitrification and sediment demand.
pH and fecal coliform influence the combined models' BOD predictions; no
unsupported direct pH/fecal-to-rate formula is invented. Oxygen-exhaustion
scenarios are unsupported. Current measured DO is needed for this optional
simulator even when the upstream BOD model itself is satellite-only.

**Presentation wording:** “We implemented a small physics-informed oxygen-balance
simulator using differential-equation losses, and checked its numerical solution
against an analytical reference. Its environmental parameters are assumed, so
it demonstrates the method rather than validated future river oxygen levels.”

Do not claim that the all-components BOD predictor now trains with these oxygen
equations. Its existing simple negative-BOD constraint remains part of that
predictor; the new equation-based PINN is an additional downstream module.

## Run

Run from `C:\Users\Lakshyajeet\Desktop\clone of dbs project\unseen-model-evaluation`:

```powershell
& 'C:\Users\Lakshyajeet\Desktop\clone of dbs project\River-Water-Quality-Prediction\.venv-check\Scripts\python.exe' scripts/explain_three_models.py
& 'C:\Users\Lakshyajeet\Desktop\clone of dbs project\River-Water-Quality-Prediction\.venv-check\Scripts\python.exe' scripts/run_oxygen_pinn.py
```

Dependencies are the existing demonstrations (SHAP/matplotlib) and experimental
(CPU PyTorch) requirements. These commands regenerate their own report folders.
They preserve every existing BOD checkpoint. Artifacts include PINN weights,
training history, training initial conditions, simulated trajectories, equation-
solver metrics, SHAP values, feature rankings, charts and hashed provenance.
