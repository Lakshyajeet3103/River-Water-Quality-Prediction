"""Four-parameter MPCB-reported surface WQI, downstream of a BOD model.

Equations and endpoint conventions: docs/wqi_method.md.
Oxygen solubility: USGS QW memorandum 2011.03, equations 7-11.
This index is not a drinking-water safety certification.
"""
import math

METHOD = 'CPCB-modified NSF four-parameter surface WQI as reported by MPCB'
METHOD_SOURCE = 'https://www.mpcb.gov.in/sites/default/files/annual_report_2019_2020_04112022.pdf'
OXYGEN_SOURCE = 'https://water.usgs.gov/admin/memo/QW/qw11.03.pdf'
WEIGHTS = {'do': 0.31, 'fecal': 0.28, 'ph': 0.22, 'bod': 0.19}


def number(value, name, low, high=None):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite number')
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not math.isfinite(value) or value < low or (high is not None and value > high):
        raise ValueError(f'{name} outside supported range [{low}, {high if high is not None else "infinity"}]')
    return value


def oxygen_solubility_mg_l(temperature_c, pressure_atm, salinity_ppt):
    """Benson-Krause equilibrium oxygen, including vapor/nonideal corrections."""
    t = number(temperature_c, 'temperature_c', 0, 40)
    p = number(pressure_atm, 'pressure_atm', 0.5, 1.1)
    s = number(salinity_ppt, 'salinity_ppt', 0, 40)
    kelvin = t + 273.15
    baseline = math.exp(-139.34411 + 1.575701e5 / kelvin - 6.642308e7 / kelvin**2
                        + 1.243800e10 / kelvin**3 - 8.621949e11 / kelvin**4)
    salinity_factor = math.exp(-s * (0.017674 - 10.754 / kelvin + 2140.7 / kelvin**2))
    vapor = math.exp(11.8571 - 3840.70 / kelvin - 216961 / kelvin**2)
    theta = 0.000975 - 1.426e-5 * t + 6.436e-8 * t**2
    pressure_factor = (p - vapor) * (1 - theta * p) / ((1 - vapor) * (1 - theta))
    return baseline * salinity_factor * pressure_factor


def oxygen_percent_saturation(do_mg_l, temperature_c, pressure_atm, salinity_ppt):
    do = number(do_mg_l, 'do_mg_l', 0)
    return 100 * do / oxygen_solubility_mg_l(temperature_c, pressure_atm, salinity_ppt)


def category(score):
    score = number(score, 'wqi', 0, 100)
    if score >= 63:
        return 'Good-Excellent'
    if score >= 50:
        return 'Good'
    if score > 38:
        return 'Poor'
    return 'Very Poor'


def calculate_wqi(*, bod_mg_l, ph, fecal_per_100ml, do_percent_saturation):
    """Calculate with supplied saturation; caller must record its provenance.

    Source rounding anomalies are retained and flagged, never silently clipped.
    The ground fecal MPN estimate may be supplied as the reported concentration;
    this does not establish equivalence of laboratory methods.
    """
    bod = number(bod_mg_l, 'bod_mg_l', 0)
    ph = number(ph, 'ph', 0, 14)
    fecal = number(fecal_per_100ml, 'fecal_per_100ml', 1)
    do = number(do_percent_saturation, 'do_percent_saturation', 0, 140)
    qdo = (0.18 + 0.66 * do if do <= 40 else -13.55 + 1.17 * do
           if do <= 100 else 163.34 - 0.62 * do)
    qfecal = (97.2 - 26.6 * math.log10(fecal) if fecal <= 1000 else
              42.33 - 7.75 * math.log10(fecal) if fecal <= 100000 else 2.0)
    if ph < 2 or ph > 12:
        qph = 0.0
    elif ph <= 5:
        qph = 16.1 + 7.35 * ph
    elif ph <= 7.3:
        qph = -142.67 + 33.5 * ph
    elif ph <= 10:
        qph = 316.96 - 29.85 * ph
    else:
        qph = 96.17 - 8 * ph
    qbod = 96.67 - 7 * bod if bod <= 10 else 38.9 - 1.23 * bod if bod <= 30 else 2.0
    q = {'do': qdo, 'fecal': qfecal, 'ph': qph, 'bod': qbod}
    contributions = {key: WEIGHTS[key] * value for key, value in q.items()}
    score = sum(contributions.values())
    warnings = [f'{key} sub-index outside 0-100 from published rounded equation'
                for key, value in q.items() if not 0 <= value <= 100]
    if not 0 <= score <= 100:
        warnings.append('Aggregate outside 0-100; no category assigned')
    return {'wqi': score, 'category': category(score) if 0 <= score <= 100 else None,
            'sub_indices': q, 'weighted_contributions': contributions,
            'formula_warnings': warnings, 'method': METHOD}


def calculate_scenario_wqi(*, bod_mg_l, ph, fecal_per_100ml, do_mg_l,
                           assumed_temperature_c, assumed_pressure_atm, assumed_salinity_ppt):
    """Explicitly assumption-based wrapper; has no hidden condition defaults."""
    oxygen = oxygen_solubility_mg_l(assumed_temperature_c, assumed_pressure_atm, assumed_salinity_ppt)
    saturation = oxygen_percent_saturation(do_mg_l, assumed_temperature_c,
                                           assumed_pressure_atm, assumed_salinity_ppt)
    result = calculate_wqi(bod_mg_l=bod_mg_l, ph=ph, fecal_per_100ml=fecal_per_100ml,
                           do_percent_saturation=saturation)
    return {**result, 'calculation_type': 'assumption-based WQI scenario',
            'assumed_temperature_c': float(assumed_temperature_c),
            'assumed_pressure_atm': float(assumed_pressure_atm),
            'assumed_salinity_ppt': float(assumed_salinity_ppt),
            'equilibrium_do_mg_l': oxygen, 'do_percent_saturation': saturation,
            'do_saturation_provenance': 'Calculated from measured DO and assumed conditions'}
