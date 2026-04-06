"""Tests for fidelity.py."""

from imads_hpo.fidelity import EpochFidelity


def test_tau_levels_descending():
    f = EpochFidelity(min_epochs=5, max_epochs=100, num_seeds=3)
    taus = f.tau_levels
    assert taus == sorted(taus, reverse=True)
    assert taus[-1] == 1  # TRUTH is always tau=1


def test_tau_to_epochs():
    f = EpochFidelity(min_epochs=5, max_epochs=100, num_seeds=3)
    assert f.tau_to_epochs(1) == 100  # TRUTH
    assert f.tau_to_epochs(20) == 5   # loose
    assert f.tau_to_epochs(10) == 10


def test_smc_levels():
    f = EpochFidelity(min_epochs=5, max_epochs=100, num_seeds=3)
    assert f.smc_levels == [1, 3]


def test_smc_levels_single_seed():
    f = EpochFidelity(min_epochs=5, max_epochs=100, num_seeds=1)
    assert f.smc_levels == [1]


def test_resolve_fidelity():
    f = EpochFidelity(min_epochs=5, max_epochs=100, num_seeds=3)
    fid = f.resolve_fidelity(tau=10, smc=3, k=2)
    assert fid.epochs == 10
    assert fid.seed_index == 2
    assert fid.tau == 10
    assert fid.smc == 3


def test_tau_levels_cover_min_and_max():
    f = EpochFidelity(min_epochs=10, max_epochs=80, num_seeds=1)
    taus = f.tau_levels
    # Loosest tau should map to at most min_epochs
    assert f.tau_to_epochs(taus[0]) >= f.min_epochs
    # Tightest tau (=1) maps to max_epochs
    assert f.tau_to_epochs(1) == f.max_epochs
