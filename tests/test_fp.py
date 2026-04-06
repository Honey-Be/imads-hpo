"""Tests for fp.py: curry and Run monad."""

from imads_hpo.fp import Run, RunLog, ask, curry, get_state, modify_state, put_state, sequence, tell


def test_curry_full_application():
    @curry
    def add(a, b, c):
        return a + b + c

    assert add(1, 2, 3) == 6


def test_curry_partial_application():
    @curry
    def add(a, b, c):
        return a + b + c

    assert add(1)(2)(3) == 6
    assert add(1, 2)(3) == 6
    assert add(1)(2, 3) == 6


def test_curry_kwargs():
    @curry
    def greet(first, last):
        return f"{first} {last}"

    assert greet(first="A")(last="B") == "A B"
    assert greet("A", last="B") == "A B"


def test_run_pure():
    state, value, log = Run.pure(42).execute("env", "state")
    assert value == 42
    assert state == "state"
    assert len(log) == 0


def test_run_map():
    state, value, log = Run.pure(10).map(lambda x: x * 2).execute("env", "s")
    assert value == 20


def test_run_bind():
    prog = Run.pure(5).bind(lambda x: Run.pure(x + 1))
    state, value, log = prog.execute("env", "s")
    assert value == 6


def test_run_then():
    prog = Run.pure(1).then(Run.pure(2))
    _, value, _ = prog.execute("env", "s")
    assert value == 2


def test_ask():
    _, value, _ = ask().execute("my_env", "state")
    assert value == "my_env"


def test_state_operations():
    prog = put_state(100).then(get_state())
    state, value, _ = prog.execute("env", 0)
    assert state == 100
    assert value == 100


def test_modify_state():
    prog = modify_state(lambda s: s + 1).then(get_state())
    state, value, _ = prog.execute("env", 10)
    assert state == 11
    assert value == 11


def test_tell():
    _, _, log = tell({"event": "hello"}).execute("env", "s")
    assert len(log) == 1
    assert log.events[0]["event"] == "hello"


def test_sequence():
    progs = [Run.pure(i) for i in range(3)]
    _, values, _ = sequence(progs).execute("env", "s")
    assert values == [0, 1, 2]


def test_run_log_accumulation():
    prog = tell({"a": 1}).then(tell({"b": 2}))
    _, _, log = prog.execute("env", "s")
    assert len(log) == 2


def test_runlog_extend():
    log = RunLog()
    log = log.extend([{"a": 1}, {"b": 2}])
    assert len(log) == 2
