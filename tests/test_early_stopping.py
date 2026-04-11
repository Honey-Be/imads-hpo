"""Tests for early stopping rules."""

from imads_hpo.early_stopping import MedianStopper, PatientStopper, ThresholdStopper


class TestPatientStopper:
    def test_no_stop_while_improving(self):
        stopper = PatientStopper(patience=3)
        # Monotonically decreasing metrics should not trigger stop.
        for i in range(10):
            assert not stopper.should_stop(i, 10.0 - i, [])

    def test_stops_after_patience(self):
        stopper = PatientStopper(patience=3)
        stopper.should_stop(0, 1.0, [])  # new best
        stopper.should_stop(1, 2.0, [])  # worse, wait=1
        stopper.should_stop(2, 2.0, [])  # worse, wait=2
        assert stopper.should_stop(3, 2.0, [])  # worse, wait=3 >= patience

    def test_resets_on_improvement(self):
        stopper = PatientStopper(patience=3)
        stopper.should_stop(0, 1.0, [])
        stopper.should_stop(1, 2.0, [])  # wait=1
        stopper.should_stop(2, 2.0, [])  # wait=2
        stopper.should_stop(3, 0.5, [])  # new best, wait=0
        assert not stopper.should_stop(4, 0.6, [])  # wait=1

    def test_min_delta(self):
        stopper = PatientStopper(patience=2, min_delta=0.1)
        stopper.should_stop(0, 1.0, [])
        # Improvement smaller than min_delta doesn't count.
        stopper.should_stop(1, 0.95, [])  # delta=0.05 < 0.1, wait=1
        assert stopper.should_stop(2, 0.95, [])  # wait=2 >= patience

    def test_reset(self):
        stopper = PatientStopper(patience=2)
        stopper.should_stop(0, 1.0, [])
        stopper.should_stop(1, 2.0, [])
        stopper.reset()
        assert stopper.best == float("inf")
        assert stopper.wait == 0


class TestMedianStopper:
    def test_no_stop_before_min_epochs(self):
        stopper = MedianStopper(
            completed_curves=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            min_epochs=5,
        )
        assert not stopper.should_stop(3, 100.0, [])

    def test_no_stop_with_insufficient_curves(self):
        stopper = MedianStopper(completed_curves=[[1.0] * 10], min_epochs=1)
        assert not stopper.should_stop(5, 100.0, [])  # only 1 curve

    def test_stops_when_above_median(self):
        stopper = MedianStopper(
            completed_curves=[
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ],
            min_epochs=1,
        )
        assert stopper.should_stop(3, 100.0, [])  # 100 > median of [4.0, 4.0]

    def test_continues_when_below_median(self):
        stopper = MedianStopper(
            completed_curves=[
                [10.0, 20.0, 30.0],
                [10.0, 20.0, 30.0],
            ],
            min_epochs=1,
        )
        assert not stopper.should_stop(1, 1.0, [])  # 1.0 < median of [20.0, 20.0]


class TestThresholdStopper:
    def test_stops_above_threshold(self):
        stopper = ThresholdStopper(threshold=5.0, min_epochs=1)
        assert stopper.should_stop(1, 10.0, [])

    def test_continues_below_threshold(self):
        stopper = ThresholdStopper(threshold=5.0, min_epochs=1)
        assert not stopper.should_stop(1, 3.0, [])

    def test_respects_min_epochs(self):
        stopper = ThresholdStopper(threshold=5.0, min_epochs=5)
        assert not stopper.should_stop(3, 100.0, [])
        assert stopper.should_stop(5, 100.0, [])
