"""Selectable calibration versions (PART A): the versions/ files, the use_calibration /
available_calibrations / $KAVIER_CALIBRATION selector, and the contract that the accessor stays
import-light (no scipy/sklearn/numpy/pandas) regardless of which version is selected."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import kavier.sdk.training.calibration as cal

SRC = Path(__file__).resolve().parents[2] / "src"
CAL_DIR = Path(cal.__file__).resolve().parent
ROOT_JSON = CAL_DIR / "calibration.json"
VERSIONS_DIR = CAL_DIR / "versions"
V4 = VERSIONS_DIR / "calibration_4model.json"
V6 = VERSIONS_DIR / "calibration_6model.json"

# Spec facts (independent of the accessor code): the two from-scratch fits cover these exact model
# sets. Sourced from the calibration design (4 dense models; 6 = those four + two extra granites).
DENSE_4 = {"mistral-7b-v0.1", "granite-3.3-8b", "granite-3-8b", "llama3.2-3b"}
ALL_6 = DENSE_4 | {"granite-3.1-2b", "granite-3.1-8b-instruct"}
_REQUIRED_KEYS = (
    "comm_scale",
    "mfu_multiplier",
    "multi_gpu_correction",
    "method_scale",
    "model_scale",
    "interaction_scale",
)
# The engine's _is_physical band (calibration/engine.py::_SCALE_LO/_SCALE_HI). Re-declared here as an
# independent invariant a shipped fit must satisfy, not imported from the code under test.
_SCALE_LO, _SCALE_HI = 0.5, 2.0


@pytest.fixture(autouse=True)
def _restore_cal():
    """Each test may swap the module global _CAL; snapshot and restore it."""
    saved = cal._CAL
    yield
    cal._CAL = saved


# --- the shipped version files ------------------------------------------------------------------


def test_versions_shipped_via_importlib_resources():
    # Packaging contract: the accessor loads versions via importlib.resources (the wheel path), not a
    # source-tree filesystem path. Falsifier: dropping versions/ from package data -> is_file() False.
    from importlib.resources import files

    for name in ("calibration_4model.json", "calibration_6model.json"):
        res = files("kavier.sdk.training").joinpath("calibration", "versions", name)
        assert res.is_file(), f"{name} not discoverable as package data"


@pytest.mark.parametrize("path", [V4, V6])
def test_version_files_valid_json_with_required_keys(path):
    # Schema contract: each shipped fit is a JSON object carrying every table the getters read.
    # Falsifier: delete "method_scale" from a version file -> missing is non-empty -> red.
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    assert not missing, f"{path.name} missing {missing}"


def test_root_calibration_is_byte_identical_to_6model():
    # CLAUDE.md contract: the shipped default (calibration.json) IS the 6-model from-scratch fit.
    # Falsifier: edit one digit in either file -> byte comparison red.
    assert ROOT_JSON.read_text(encoding="utf-8") == V6.read_text(encoding="utf-8")


def test_6model_covers_all_six_models():
    # Oracle: the six model names of the 6-model fit (spec fact above).
    # Falsifier: regenerate 6model dropping granite-3.1-2b -> set inequality red.
    data = json.loads(V6.read_text(encoding="utf-8"))
    assert set(data["model_scale"]) == ALL_6


def test_4model_covers_exactly_dense_four():
    # Oracle: the four dense model names (spec fact). The 4-model fit must not leak any 5th/6th model
    # into model_scale OR into the interaction_scale cell keys.
    # Falsifier: a stray "granite-3.1-2b|..." interaction key -> the `all(...)` guard red.
    data = json.loads(V4.read_text(encoding="utf-8"))
    assert set(data["model_scale"]) == DENSE_4
    assert all(k.split("|")[0] in DENSE_4 for k in data["interaction_scale"])


@pytest.mark.parametrize("path", [V4, V6])
def test_shipped_fits_are_physical(path):
    # Invariant the engine's _is_physical guard enforces at fit time: comm_scale and every per-GPU
    # mfu_multiplier lie in the physical band [0.5, 2.0]. Falsifier: a hand-edited comm_scale of 3.0
    # (a non-identifiable / noise-fitting fit) -> out of band -> red.
    data = json.loads(path.read_text(encoding="utf-8"))
    assert _SCALE_LO <= data["comm_scale"] <= _SCALE_HI
    assert all(_SCALE_LO <= v <= _SCALE_HI for v in data["mfu_multiplier"].values())


# --- the selector API ---------------------------------------------------------------------------


def test_available_calibrations_is_default_plus_sorted_versions():
    # Exact contract: "default" first, then one name per versions/calibration_<name>.json, sorted.
    # 4model = thesis validation, 6model = thesis exploration, allmodels = the non-thesis all-model fit.
    # Falsifier: a spurious/missing version file, or wrong order (allmodels sorts after the digits) -> red.
    assert cal.available_calibrations() == ["default", "4model", "6model", "allmodels"]


def test_use_calibration_by_name_switches_live_table():
    # Switching by name must install that fit as the live table AND the getters must deref it live.
    # comm_scale oracle read straight from the file (independent of the accessor).
    cal.use_calibration("4model")
    assert set(cal._active_calibration()["model_scale"]) == DENSE_4
    assert cal.get_comm_scale() == pytest.approx(json.loads(V4.read_text())["comm_scale"])
    # Falsifier: if use_calibration cached the first table, this second switch's model set stays 4.
    cal.use_calibration("6model")
    assert set(cal._active_calibration()["model_scale"]) == ALL_6
    assert cal.get_comm_scale() == pytest.approx(json.loads(V6.read_text())["comm_scale"])


def test_use_calibration_default_alias_resolves_to_root_file():
    # The "default" alias must resolve to calibration.json, not to a version file.
    # Falsifier: aliasing "default" to 4model -> comm_scale mismatch (root != V4 comm_scale).
    cal.use_calibration("default")
    assert cal.get_comm_scale() == pytest.approx(json.loads(ROOT_JSON.read_text())["comm_scale"])


def test_use_calibration_by_path():
    # A filesystem path (has .json / os.sep) takes the path branch, not the name branch.
    # Falsifier: path-detection misfires -> "unknown calibration" ValueError instead of loading.
    loaded = cal.use_calibration(str(V4))
    assert set(loaded["model_scale"]) == DENSE_4


def test_use_calibration_unknown_name_raises_value_error():
    # Falsifier: falling back to default silently instead of raising -> no exception -> red.
    with pytest.raises(ValueError, match="unknown calibration"):
        cal.use_calibration("does-not-exist")


def test_use_calibration_missing_path_raises_file_not_found():
    # A name that looks like a path but does not exist must raise, not snap to a version.
    # Falsifier: swallowing the missing file and returning default -> no exception -> red.
    with pytest.raises(FileNotFoundError):
        cal.use_calibration("/no/such/calibration.json")


def test_use_calibration_returns_the_installed_table_object():
    # Identity contract: the returned dict IS the live _CAL, not a copy.
    # Falsifier: returning a deepcopy while setting _CAL to another object -> `is` red.
    out = cal.use_calibration("4model")
    assert out is cal._CAL
    assert set(out["model_scale"]) == DENSE_4


# --- env var + import-light contract (fresh interpreters) ---------------------------------------


def _subproc(code: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ}
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update(extra_env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


def test_env_var_selects_calibration_on_first_access():
    # $KAVIER_CALIBRATION picks the table loaded on first access. Oracle: the 4- vs 6-model counts.
    # Falsifier: ignoring the env var -> "default" (6) printed for every case, so 4model -> "6".
    code = "import kavier.sdk.training.calibration as c; print(len(c._active_calibration()['model_scale']))"
    assert _subproc(code, {"KAVIER_CALIBRATION": "4model"}).stdout.strip() == "4"
    assert _subproc(code, {"KAVIER_CALIBRATION": "6model"}).stdout.strip() == "6"
    assert _subproc(code, {"KAVIER_CALIBRATION": "default"}).stdout.strip() == "6"


def test_default_without_env_is_six_model():
    # With no env var set, first access falls back to the root (6-model) file via `... or "default"`.
    # Falsifier: a fallback to 4model -> "4"; a crash on the missing var -> non-zero returncode.
    code = "import kavier.sdk.training.calibration as c; print(len(c._active_calibration()['model_scale']))"
    env = {k: v for k, v in os.environ.items() if k != "KAVIER_CALIBRATION"}
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "6"


@pytest.mark.parametrize("env", [{}, {"KAVIER_CALIBRATION": "4model"}])
def test_accessor_import_is_stdlib_only(env):
    # Import-light contract: importing + using the accessor (any selected version) must not drag in the
    # heavy fit deps. Falsifier: __init__ importing engine.py -> numpy/scipy in sys.modules -> non-[].
    code = (
        "import sys, kavier.sdk.training.calibration as c; "
        "c.get_comm_scale(); "
        "print([m for m in ('scipy','sklearn','numpy','pandas') if m in sys.modules])"
    )
    proc = _subproc(code, env or None)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout
