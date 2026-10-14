# WQI method selection (step 1)

Status: method selected and documented. [Step 3 calculator](wqi_calculator.md) implements explicitly assumption-based scenarios, and [step 4 evaluation](wqi_evaluation.md) compares the original 20 test cases; existing BOD models remain unchanged. Notebook integration remains pending.

Step 2 result: [input audit](wqi_input_audit/README.md) found complete BOD/pH/DO/fecal values, but zero matched measured temperatures. The selected method needs additional data or an explicitly labelled oxygen-conversion scenario before implementation can produce scores.

## Selected method and source

Use the **four-parameter CPCB-modified NSF surface-water index as reported by MPCB**, not the original nine-parameter NSF index or a groundwater index.

Primary reference: [MPCB Annual Report 2019-20](https://www.mpcb.gov.in/sites/default/files/annual_report_2019_2020_04112022.pdf), section 5.4.1.1, Tables 5.26-5.28 (PDF pages 37-38, printed pages 35-36). Verified on 2026-10-14.

This is a published method being applied to our observations, not an official CPCB certification of project predictions.

## Inputs and aggregation

| Input | Required unit | Weight |
|---|---|---:|
| DO | Percent saturation | 0.31 |
| Fecal coliform (FC) | Counts/100 mL; verify compatibility with source MPN/100 mL | 0.28 |
| pH | Dimensionless | 0.22 |
| BOD | mg/L | 0.19 |

`WQI = 0.31*qDO + 0.28*qFC + 0.22*qpH + 0.19*qBOD`

Here q denotes each parameter's sub-index. Higher WQI indicates better quality under this index.

## Published sub-index equations

In each row, x is that parameter in the required unit; log means base 10.

| Parameter | Published range | q(x) |
|---|---|---|
| DO | 0-40 | 0.18 + 0.66*x |
| DO | 40-100 | -13.55 + 1.17*x |
| DO | 100-140 | 163.34 - 0.62*x |
| FC | 1-10^3 | 97.2 - 26.6*log10(x) |
| FC | 10^3-10^5 | 42.33 - 7.75*log10(x) |
| FC | >10^5 | 2 |
| pH | 2-5 | 16.1 + 7.35*x |
| pH | 5-7.3 | -142.67 + 33.5*x |
| pH | 7.3-10 | 316.96 - 29.85*x |
| pH | 10-12 | 96.17 - 8*x |
| pH | <2 or >12 | 0 |
| BOD | 0-10 | 96.67 - 7*x |
| BOD | 10-30 | 38.9 - 1.23*x |
| BOD | >30 | 2 |

Source category thresholds: 63, 50, 38; labels respectively Good-Excellent, Good water, Poor Water, Very Poor water.

## Project implementation conventions

The printed ranges overlap at endpoints and the rounded equations have small discontinuities. To make later code reproducible, use the lower-range equation at shared endpoints (DO 40/100, FC 1,000/100,000, pH 5/7.3/10, BOD 10/30). Do not claim that this tie convention is specified by the source.

Use unrounded scores for classification: WQI >=63 Good-Excellent; >=50 Good; >38 Poor; <=38 Very Poor. This assigns overlapping printed endpoints explicitly. Display rounding must not change classification. Never classify a missing score.

Require finite values. Reject negative BOD, FC below 1, pH outside 0-14, and DO saturation outside 0-140 until a documented extension is agreed. Do not extrapolate missing source ranges, replace censored values, drop parameters, change weights, or silently invent inputs. Inspect computed sub-indices and flag out-of-range results; do not silently clamp away source-equation anomalies.

## Oxygen conversion requirement

Our enriched dataset contains `ground_dissolved_oxygen` in mg/L, not percent saturation. These units cannot be substituted for each other.

`DO_percent_saturation = 100 * measured_DO_mg_L / equilibrium_DO_mg_L`

Equilibrium oxygen depends on water temperature, atmospheric pressure and salinity/conductance. [USGS DOTABLES](https://www.usgs.gov/tools/dotables) documents these inputs and provides oxygen-solubility and saturation calculations. During step 2, look for matched water temperature and supporting conditions or directly measured saturation. Record any justified approximation explicitly; a constant oxygen denominator must not be presented as measured saturation. Satellite land-surface temperature is not automatically river-water temperature.

The current prepared dataset does not establish these additional conversion inputs. Therefore selecting this method does not yet establish that all 100 observations can receive a defensible WQI.

## How it will attach to the saved model

Estimated WQI will use saved-model predicted BOD plus measured pH, DO and fecal coliform. Reference WQI will use measured BOD with exactly the same remaining inputs and calculation. Both need the same documented oxygen conversion.

Keep WQI as a downstream calculation; no retraining or model-input changes. Preserve original artifacts. Label outputs **estimated four-parameter WQI using predicted BOD and measured station parameters**. Coordinates alone are insufficient.

Later comparison on the existing 20 test observations measures the effect of substituting predicted BOD into the formula. It is not independent validation of all four predicted parameters, a drinking-water safety assessment, or proof of monitoring unseen river stretches. Full data audit, implementation, tests and notebook outputs belong to subsequent steps.
