"""Tests for repro.py: determinism and seed derivation."""

from imads_hpo.repro import SeedPath, derive_seed


def test_derive_seed_deterministic():
    s1 = derive_seed(42, "train", 0, None, None, None)
    s2 = derive_seed(42, "train", 0, None, None, None)
    assert s1 == s2


def test_derive_seed_different_namespace():
    s1 = derive_seed(42, "train", 0, None, None, None)
    s2 = derive_seed(42, "eval", 0, None, None, None)
    assert s1 != s2


def test_derive_seed_curried():
    # derive_seed has defaults for rank/worker/epoch/step, so
    # curry(derive_seed)(master_seed)(namespace) executes immediately.
    make = derive_seed(42)
    s1 = make("train")  # executes with defaults: rank=0, worker=None, epoch=None, step=None
    s2 = derive_seed(42, "train")
    assert s1 == s2


def test_seed_path_deterministic():
    p1 = SeedPath(42).child("trial", 7).child("sample", 0)
    p2 = SeedPath(42).child("trial", 7).child("sample", 0)
    assert p1.seed() == p2.seed()


def test_seed_path_different_children():
    p1 = SeedPath(42).child("trial", 7).child("sample", 0)
    p2 = SeedPath(42).child("trial", 7).child("sample", 1)
    assert p1.seed() != p2.seed()


def test_seed_path_different_master():
    p1 = SeedPath(42).child("a")
    p2 = SeedPath(43).child("a")
    assert p1.seed() != p2.seed()
