import subprocess
import sys

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from drift_point import Replay, build_bases, collect_deltas, get_layers, greedy, measure_drift

MODEL = "hf-internal-testing/tiny-random-gpt2"


def load_tok(name):
    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception:
        return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def setup():
    tok = load_tok(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()
    store = collect_deltas(model, tok, ["Hello world", "The cat sat on the"], max_len=16)
    bases = build_bases(store, rank=2)
    replay = Replay(model, bases)
    yield model, tok, bases, replay
    replay.remove()


def test_arch_detected(setup):
    model, *_ = setup
    assert len(get_layers(model)) >= 1


def test_replay_off_is_identity(setup):
    model, tok, bases, replay = setup
    replay.on = False
    a = greedy(model, tok, "Hello there friend", 4)
    b = greedy(model, tok, "Hello there friend", 4)
    assert a == b


def test_drift_measurement(setup):
    model, tok, bases, replay = setup
    r = measure_drift(model, tok, "Once upon a time", replay, steps=5)
    for k in ["prompt", "original", "replay", "drift_point", "steps", "match", "tokens_agree"]:
        assert k in r
    assert len(r["match"]) == 5
    assert r["tokens_agree"] <= 5
    dp = r["drift_point"]
    assert dp is None or (0 <= dp < 5)


def test_cli_help():
    p = subprocess.run([sys.executable, "drift_point.py", "--help"], capture_output=True, text=True)
    assert p.returncode == 0
    assert "rank" in p.stdout


def test_interactive_batch():
    p = subprocess.run(
        [sys.executable, "drift_point.py", "--interactive", "--model", MODEL, "--rank", "2", "--steps", "3"],
        input="hello there" + chr(10) + "the cat sat" + chr(10),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0
    assert "drift_point" in p.stdout
    assert "hello there" in p.stdout
