import pandas as pd
import pytest

from river_water_quality.training.modality_explanation import explain_groups


def test_linear_attribution_uses_only_input_groups():
    background = pd.DataFrame({'s': [1., 3.], 'g': [2., 4.], 'ground_bod': [999., 999.]})
    example = pd.Series({'s': 5., 'g': 1., 'ground_bod': -999.})
    result = explain_groups(lambda x: (2 * x.s + 3 * x.g).to_numpy(), example, background, ['s'], ['g'])
    assert result['satellite_contribution_mg_l'] == pytest.approx(6)
    assert result['ground_contribution_mg_l'] == pytest.approx(-6)
    assert result['predicted_bod_mg_l'] == pytest.approx(13)
    assert result['reconstruction_error_mg_l'] == pytest.approx(0)


def test_interaction_shared_and_invalid_groups_rejected():
    background = pd.DataFrame({'s': [0.], 'g': [0.]})
    example = pd.Series({'s': 2., 'g': 3.})
    result = explain_groups(lambda x: (x.s * x.g).to_numpy(), example, background, ['s'], ['g'])
    assert result['satellite_contribution_mg_l'] == result['ground_contribution_mg_l'] == 3
    with pytest.raises(ValueError):
        explain_groups(lambda x: x.s.to_numpy(), example, background, ['s'], ['s'])
