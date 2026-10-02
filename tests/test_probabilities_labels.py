"""The collector's decision-label rows: kept inside the collected range, stable across replay, finite."""

import math

import pytest

from tensorfold.engine.probabilities import Probabilities


def test_label_rows_keep_only_the_collected_positions():
    record = Probabilities(0, 10, 1, labels=[4, 9])
    record.add_labels([9, 10, 11], [[-1.0, -2.0], [-0.5, -1.5], [-3.0, -4.0]])
    assert record.label_rows == {10: [-0.5, -1.5]}
    assert record.labels == (4, 9)


def test_a_replay_must_give_the_same_label_row():
    record = Probabilities(0, 0, 1, labels=[1])
    record.add_labels([0], [[-0.25]])
    record.add_labels([0], [[-0.25]])
    with pytest.raises(RuntimeError, match="replay"):
        record.add_labels([0], [[-0.5]])


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_non_finite_label_rows_are_refused(value):
    record = Probabilities(0, 0, 1, labels=[1, 2])
    with pytest.raises(RuntimeError, match="finite"):
        record.add_labels([0], [[-1.0, value]])


def test_no_labels_by_default():
    record = Probabilities(5, 0, 3)
    assert record.labels == () and record.label_rows == {}
