# Model inventory and presentation recommendation — 14 October 2026

Audited published Git branches after fetching all remote heads, especially
`main` (6575b88), `bod-categories-crossregion-2026-10-14` (fcb8f31),
`working-model-complete`, `feature/hybridization-fusion-poc`, and
`friend-testing-data-2026-10-13`. No models were retrained for this inventory.

## Count and terminology

Across all remote branches there are **57 distinct committed checkpoint-file
contents** (.joblib/.pt/.pth/.pkl, deduplicated by Git blob ID). This is not 57
independent model designs. Of these, 37 belong to station-holdout evaluations:
9 original three-station fits, 21 expanded seven-station fits, and 7 nested
band-selection fits. The other 20 include ordinary experiments and repeated
versions. Some experiments have published metrics/code but no committed weights.
Temporary software-test models and tuning candidates are not additional products.

MAE is average absolute error: lower is better. BOD errors below are in mg/L,
not percentages. Except for the explicit category/WQI and synthetic tables,
all outputs are predicted BOD. Measured BOD is the training/testing answer,
never a same-event predictor. FC means measured fecal coliform; DO means
dissolved oxygen. Ten bands means B2/B3/B4/B5/B6/B7/B8/B8A/B11/B12 surface
reflectance means, not image patches or CNN inputs.

## 1. Main familiar-station comparison

These rows use 80 training events and the identical 20 later test events from
Anupshahar, Varanasi and Bijnor. These are familiar locations, not training rows.
The test set was examined repeatedly: results are exploratory.

| Model/version | Prediction inputs | Output | Train / test | MAE | Interpretation |
|---|---|---|---|---:|---|
| Original satellite MLP | Ten bands | BOD | 80 / 20 | 0.8225 | Historical reference |
| Selected satellite + ratios MLP | Ten bands; five ratios calculated internally | BOD | 80 / 20 | 0.5911 | Recommended satellite-only demo |
| Ground-only MLP | FC, pH, DO | BOD | 80 / 20 | 0.2484 | Strong ground-data reference |
| Satellite + FC MLP | Ten bands + FC | BOD | 80 / 20 | 0.3268 | Compatible with saved US inputs |
| All-input combined MLP | Ten bands + FC, pH, DO | BOD | 80 / 20 | 0.3173 | SHAP demonstration available |
| Selected-feature combined MLP | B8, B8A, B11 + FC | BOD | 80 / 20 | **0.1852** | Lowest error in this comparison |
| Two-branch network: switches off | Ten bands + FC, pH, DO | BOD | 80 / 20 | 0.3211 | Experimental reference |
| Two-branch: gate only | Same + sample/image date gap | BOD | 80 / 20 | **0.2799** | Best of these five network variants |
| Two-branch: constraint only | Ten bands + FC, pH, DO | BOD | 80 / 20 | 0.3221 | Little change |
| Two-branch: station adversarial only | Ten bands + FC, pH, DO | BOD | 80 / 20 | 0.3252 | Slightly worse |
| Two-branch: all three switches on | Ten bands + FC, pH, DO + date gap | BOD | 80 / 20 | **0.2827** | Best architecture/features demonstration |
| Training-average reference | No per-event inputs | BOD | 80 / 20 | 1.0054 | Simple benchmark |

Station identities train the adversarial head but are not inference predictors.
The constraint penalizes negative BOD, not a full physics-equation model. These
two-branch networks use small dense layers, not CNNs. Ordinary combined MLPs
concatenate inputs; they are not learned dual-branch gated architectures.

Sources: [main comparison](https://github.com/Lakshyajeet3103/River-Water-Quality-Prediction/blob/bod-categories-crossregion-2026-10-14/demo_bundle/files/outputs/straightforward_demos_20261014/metrics.csv),
[ground reference](https://github.com/Lakshyajeet3103/River-Water-Quality-Prediction/blob/bod-categories-crossregion-2026-10-14/demo_bundle/files/outputs/ground_only_comparison_20261014/metrics.csv),
[switchable network](https://github.com/Lakshyajeet3103/River-Water-Quality-Prediction/blob/bod-categories-crossregion-2026-10-14/docs/experimental_network.md).

## 2. Other satellite refinements

All predict BOD at familiar stations. These trials do not prove performance at
new stations. The 19-case rows below use an identical restricted test cohort.

| Trial | Inputs/settings | Train | Test | MAE |
|---|---|---:|---:|---:|
| Visible-band selection | B2/B3/B4 | 80 | 20 | 1.0961 |
| Regularisation only | Ten bands; validation retained alpha=1 | 80 | 20 | 0.8225 |
| Joint ratio/settings selection | Ten bands + five ratios; alpha=0.01 | 80 | 20 | 0.8758 |
| Original on three-day-eligible test rows | Ten bands | 80 | 19 | 0.7397 |
| Selected ratio model on same eligible rows | Ten bands + five ratios; alpha=1 | 80 | 19 | 0.5660 |
| Joint ratio/settings model on eligible rows | Ten bands + five ratios; alpha=0.01 | 80 | 19 | 0.7376 |
| Retrain using three-day matches only | Ten bands; alpha=1 | 57 | 19 | 0.9001 |
| Three-day ratio/settings selection | Ten bands + five ratios; alpha=0.1 | 57 | 19 | 0.9573 |

Regularisation-only reproduces the original model; it is not a different winning
model. Three-day filtering reduces the training cohort as well as the date gap.
Source: [refinement metrics](https://github.com/Lakshyajeet3103/River-Water-Quality-Prediction/blob/bod-categories-crossregion-2026-10-14/demo_bundle/files/outputs/satellite_seen_refinements_20261014/metrics.csv).

## 3. Unseen Indian stations

All outputs are BOD. Holdout rows are pooled predictions from separately fitted
fold models. Each station is excluded from the model used to predict its rows;
shared satellite acquisitions are purged. Dates can overlap across stations.
The same 100/130 rows are reused across model comparisons, not new independent
datasets for every row in this table.

| Evaluation | Inputs/model | Test observations | MAE |
|---|---|---:|---:|
| Original three-station rotation | Ten-band satellite MLP | 100 pooled | 2.0849 |
| Original three-station rotation | FC, pH, DO MLP | 100 pooled | 1.8990 |
| Original three-station rotation | Ten bands + FC, pH, DO MLP | 100 pooled | 1.4957 |
| Original three-station rotation | Fold training average | 100 pooled | 1.6547 |
| Expanded seven-station rotation | Ten-band satellite MLP | 130 pooled | 1.7282 |
| Expanded seven-station rotation | FC, pH, DO MLP | 130 pooled | **0.8981** |
| Expanded seven-station rotation | Ten bands + FC, pH, DO MLP | 130 pooled | 1.5900 |
| Expanded seven-station rotation | Training-selected satellite subsets; 1–2 bands per fold | 130 pooled | 1.3315 |
| Expanded seven-station rotation | Fold training average | 130 pooled | 1.2435 |
| Frozen original 80-row model, four new stations | Ten-band satellite MLP | 30 | 0.9205 |
| Frozen original 80-row model, four new stations | FC, pH, DO MLP | 30 | **0.4850** |
| Frozen original 80-row model, four new stations | Ten bands + FC, pH, DO MLP | 30 | 0.8271 |
| Frozen original 80-row reference, four new stations | Training average | 30 | 0.5172 |

**New evaluation added after this inventory:** the frozen ratio model,
selected four-input model and all five experimental networks were tested on
the 30 new-station cases. Four cases sharing training satellite acquisitions
were excluded from the primary 26-case comparison. Primary MAE/MAPE:
ratio 0.9531 mg/L / 51.18%; selected four-input 0.7548 / 33.67%;
network baseline 1.1163 / 57.24%; gate 0.6925 / 33.26%; constraint
1.1176 / 57.32%; adversarial 1.0971 / 56.28%; all components 0.6937 /
33.35%. The mean reference has MAE 0.5197 / MAPE 33.46%.
All trained models have worse MAE than the mean reference. Full 30-case scores
are also saved. See the isolated `unseen-model-evaluation` workspace's
`docs/saved_models_unseen.md` and `docs/saved_models_unseen_results/`.
These are frozen checkpoints, distinct from the rotating fold models above.

Sources: [station evaluation](https://github.com/Lakshyajeet3103/River-Water-Quality-Prediction/blob/bod-categories-crossregion-2026-10-14/docs/unseen_station_evaluation.md),
[nested band selection](https://github.com/Lakshyajeet3103/River-Water-Quality-Prediction/blob/bod-categories-crossregion-2026-10-14/docs/satellite_band_selection.md).

## 4. Four real US matches

Every row predicts BOD on the same four observations at three US stations.
Measured BOD is 2, 3, 9 and 3 mg/L; these rivers are not assumed clean. The
different model generations were trained on different India cohorts.

| Frozen India-trained model | Prediction inputs | Test | MAE |
|---|---|---:|---:|
| Original six-training-row PyTorch fusion | Ten bands + FC | 4 | 3.4762 |
| Expanded 38-training-row combined MLP | Ten bands + FC | 4 | 3.8991 |
| Current 80-training-row selected ratio MLP | Ten bands + five ratios | 4 | 2.6533 |
| Current 80-training-row combined MLP | Ten bands + FC | 4 | 3.8415 |
| Current India training-average reference | No per-event inputs | 4 | 2.0994 |

The current models underperform their mean reference. This is a completed
cross-country test, not demonstrated robustness. Satellite/ground Shapley
contributions are saved for the compatible 80-row combined model. The US sample
has no pH/DO and no station-average profiles, so the full-chemistry models and
custom categories were not evaluated on it.

Sources: [current foreign report](https://github.com/Lakshyajeet3103/River-Water-Quality-Prediction/blob/bod-categories-crossregion-2026-10-14/docs/crossregion_explainable.md),
[original foreign metrics](https://github.com/Lakshyajeet3103/River-Water-Quality-Prediction/blob/bod-categories-crossregion-2026-10-14/demo_bundle/files/outputs/crossregion_demo_001/metrics.csv).
The expanded 38-row model's 3.8991 figure is also retained locally in
`outputs/crossregion_sklearn_expanded_20261014/metrics.csv`.

## 5. Category and WQI calculations (not new trained models)

| Calculation | Inputs | Output | Test population | Result |
|---|---|---|---|---|
| BOD-led custom category | Ratio-model BOD + training-only station-average pH/DO/FC | Best / Good / Bad | Same 20 familiar-station cases | 13/20 agreement = 65%; 7/20 disagreement |
| Earlier four-parameter WQI scenario | Original ten-band-model BOD + measured pH/DO/FC; assumed temperature/pressure/salinity | WQI score and four-category label | Same 20 cases at each temperature | WQI MAE 1.094 points; agreement 80% at 20 C, 90% at 25 C, 95% at 30 C |

Custom-category reference labels use the actual held-out chemistry and BOD in
the same custom formula. There are no Bad references and ten boundary-sensitive
predictions. Always predicting Best would also match 13/20, although the
training-majority and training-mean-BOD baselines match only 7/20.

For the earlier WQI, reference and prediction share measured inputs carrying
81% of the weight. Its high agreement is not independent satellite-only WQI
accuracy. Its mean-BOD baseline achieves 90% at every temperature. Neither
calculation produces a drinking-water safety certification.

Sources: [category report](https://github.com/Lakshyajeet3103/River-Water-Quality-Prediction/blob/bod-categories-crossregion-2026-10-14/docs/bod_led_category.md),
[WQI evaluation](https://github.com/Lakshyajeet3103/River-Water-Quality-Prediction/blob/bod-categories-crossregion-2026-10-14/docs/wqi_evaluation.md).

## 6. Older real-data experiments

All rows estimate BOD at familiar stations, except the explicit CPCB forecast.
These different datasets/test sizes prevent fair ranking against the 80/20 models.

| Model | Inputs | Training / validation / test | Test MAE |
|---|---|---|---:|
| Original PyTorch two-branch fusion | Ten bands + FC | 6 / 5 / 3 | 0.5864 |
| Original feature-selection rerun | Selected all ten bands + FC | 6 / 5 / 3 | 0.5864 |
| Friend's original combined MLP, displayed by frontend | Ten bands + FC | 6 / 5 / 3 | 0.6879 |
| Original ground Ridge | FC | 6 / 5 / 3 | 1.0043 |
| Original satellite Ridge | Ten bands | 6 / 5 / 3 | 0.9201 |
| Expanded combined MLP | Ten bands + FC | 38 / 5 / 17; four events purged | 0.5360 |
| Same frozen expanded combined MLP, additional dates | Ten bands + FC | Same 38 trained / 36 additional tests | 0.6462 |
| Earlier CPCB-only persistence forecast | Most recent past BOD; history eligibility rules | 1672 / 921 / 1036; 101 familiar stations | 1.1425 |
| Earlier CPCB-only random forest candidate | Last BOD, last-three mean, forecast gap, seasonal sine/cosine | 1672 / 921 validation | No saved test score; validation MAE 1.7868 |

The persistence model predicts a later reading by carrying forward the last
reading; it is a baseline, not a learned satellite model. Its saved local report
is `outputs/mvp/report.json`; no corresponding committed metrics/report was found
in the fetched remote branches. The larger test count belongs to a different task.

Sources: [friend's original model metrics](https://github.com/Lakshyajeet3103/River-Water-Quality-Prediction/blob/main/artifacts/restored-demo/outputs/fusion_sklearn_001/metrics.csv),
[original fusion](https://github.com/Lakshyajeet3103/River-Water-Quality-Prediction/blob/bod-categories-crossregion-2026-10-14/demo_bundle/files/outputs/fusion_demo_001/metrics.csv),
[expanded combined model](https://github.com/Lakshyajeet3103/River-Water-Quality-Prediction/blob/bod-categories-crossregion-2026-10-14/demo_bundle/files/outputs/fusion_sklearn_expanded_20261014/metrics.csv).

## 7. Friend's synthetic advanced-feature demo

These figures use generated numbers, not CPCB/USGS observations. There are
480 synthetic source examples and 240 synthetic target examples. DANN uses
unlabelled target features during adaptation, so its target set is not an
entirely unseen domain. Errors below are synthetic target units, not verified
real BOD mg/L. Source-only models see source labels only.

| Model | Inputs | Output | Target examples evaluated | MAE |
|---|---|---|---:|---:|
| Source-only MLP | Ten simulated satellite + three simulated ground variables | Synthetic target | 240 | 0.3426 |
| Domain-adversarial network | Same; source/target domain labels during training | Synthetic target | 240 | 0.3532 |
| Ordinary concatenated fusion | Same + two simulated observation ages | Synthetic target | 240 | 0.2458 |
| Temporal gated two-branch network | Same + two simulated observation ages | Synthetic target | 240 | 0.2704 |

The DANN and gate did not improve their respective synthetic comparisons.
Dynamic INT8 quantization was also demonstrated; this tiny model's saved state
became larger and measured batch inference became slower. It is not a separate
water-quality accuracy result.

Source: [explicit synthetic-data report](https://github.com/Lakshyajeet3103/River-Water-Quality-Prediction/blob/main/artifacts/advanced_demo_001/report.json).

## Recommendation for a short professor demonstration

1. Present the 100 real matched events, the 80/20 split, and one clear BOD target.
2. Show the ordinary comparisons and the selected-feature combined model:
   B8/B8A/B11 + measured FC, MAE 0.1852 on 20 familiar-station cases. This is
   the lowest recorded same-test error, not proof it is best everywhere.
3. Show the experimental two-branch network with all switches: MAE 0.2827,
   and the five-variant table. Gate-only improves 0.3211 to 0.2799; constraint
   and station adversarial alone did not help. This is the architecture demo.
4. Show SHAP/group explanations for the model they actually explain. Existing
   all-chemistry SHAP and satellite-plus-FC group explanations do not explain
   the selected four-input or all-components model automatically.
5. Show the selected satellite-only model (0.5911) and optional custom category
   calculation (65% agreement) as the no-current-ground-readings demonstration.
6. Finish with unseen-station/US results as evidence that transfer was tested
   and remains limited. Explain failure honestly rather than adding features.

For a single model that demonstrates the proposed architecture, choose the
**all-components experimental network**. For the strongest measured score in
the common test, choose the **four-input selected-feature MLP**. For prediction
without current ground measurements, choose the **satellite ratio MLP**.

Use the `bod-categories-crossregion-2026-10-14` branch for the comprehensive
artifacts and the earlier executed demonstration notebook for its documented
historical experiments. The frontend on `main` still reads the older three-test-
case model's saved predictions; it has not been connected to these newer models.
Notebook integration of the newest category/US additions is also pending.
