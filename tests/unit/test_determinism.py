"""Tests for the determinism controls and project path anchoring."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ml_platform import determinism, paths


class TestThreadPinning:
    def test_pinning_sets_every_backend_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for variable in determinism.THREAD_ENV_VARS:
            monkeypatch.delenv(variable, raising=False)
        determinism.pin_threads(1)
        for variable in determinism.THREAD_ENV_VARS:
            assert os.environ[variable] == "1"

    def test_verification_reports_unpinned_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        determinism.pin_threads(1)
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
        assert determinism.verify_thread_pinning(1) == ["OMP_NUM_THREADS"]

    def test_verification_is_empty_when_pinning_took_effect(self) -> None:
        determinism.pin_threads(2)
        assert determinism.verify_thread_pinning(2) == []

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_thread_counts_are_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            determinism.pin_threads(bad)


class TestSeeding:
    def test_seeding_makes_numpy_draws_repeatable(self) -> None:
        import numpy as np

        determinism.set_global_seed(42)
        first = np.random.rand(5)
        determinism.set_global_seed(42)
        assert np.array_equal(first, np.random.rand(5))

    def test_seeding_makes_python_random_repeatable(self) -> None:
        import random

        determinism.set_global_seed(7)
        first = [random.random() for _ in range(5)]
        determinism.set_global_seed(7)
        assert first == [random.random() for _ in range(5)]

    def test_hash_seed_is_recorded_in_the_environment(self) -> None:
        determinism.set_global_seed(123)
        assert os.environ["PYTHONHASHSEED"] == "123"


class TestRowOrder:
    """Row order changes the gradient boosting model, so it is pinned and hashed.

    Switching the sort from quicksort to stable was measured to move candidate
    average precision from 0.701151 to 0.702866. The sort kind is therefore a
    pinned constant, and the resulting order is fingerprinted so that any change
    fails loudly rather than drifting every metric.
    """

    def test_sort_kind_is_pinned_to_the_reference_ordering(self) -> None:
        assert determinism.ROW_SORT_KIND == "quicksort"

    def test_sorting_is_repeatable_for_identical_input(self) -> None:
        import pandas as pd

        frame = pd.DataFrame({"key": [3, 1, 2, 1, 3, 1], "id": list("abcdef")})
        first = frame.sort_values("key", kind=determinism.ROW_SORT_KIND)["id"].tolist()
        second = frame.sort_values("key", kind=determinism.ROW_SORT_KIND)["id"].tolist()
        assert first == second


class TestRowOrderFingerprint:
    @staticmethod
    def _frame(terms: list[int], targets: list[int]) -> object:
        import pandas as pd

        return pd.DataFrame(
            {
                "ApprovalDate": pd.to_datetime(["2005-01-01"] * len(terms)),
                "Term": terms,
                "target": targets,
            }
        )

    def test_identical_frames_share_a_fingerprint(self) -> None:
        a = self._frame([84, 120, 240], [0, 1, 0])
        b = self._frame([84, 120, 240], [0, 1, 0])
        assert determinism.row_order_fingerprint(a) == determinism.row_order_fingerprint(b)

    def test_reordered_rows_change_the_fingerprint(self) -> None:
        """The property that makes the fingerprint useful.

        Same content, different order. Gradient boosting would produce a
        different model, so the fingerprint must distinguish them.
        """
        a = self._frame([84, 120, 240], [0, 1, 0])
        b = self._frame([240, 120, 84], [0, 1, 0])
        assert determinism.row_order_fingerprint(a) != determinism.row_order_fingerprint(b)

    def test_changed_content_changes_the_fingerprint(self) -> None:
        a = self._frame([84, 120, 240], [0, 1, 0])
        b = self._frame([84, 120, 240], [0, 1, 1])
        assert determinism.row_order_fingerprint(a) != determinism.row_order_fingerprint(b)

    def test_missing_column_is_reported(self) -> None:
        import pandas as pd

        with pytest.raises(KeyError, match="fingerprint column"):
            determinism.row_order_fingerprint(pd.DataFrame({"Term": [84]}))


class TestCapture:
    def test_settings_record_what_is_in_force(self) -> None:
        determinism.pin_threads(1)
        settings = determinism.capture(seed=42, n_threads=1)
        assert settings.seed == 42
        assert settings.n_threads == 1
        assert settings.sort_kind == determinism.ROW_SORT_KIND
        assert settings.thread_env["OMP_NUM_THREADS"] == "1"

    def test_settings_serialise_for_the_run_record(self) -> None:
        payload = determinism.capture(seed=1, n_threads=1).to_dict()
        assert set(payload) == {"seed", "n_threads", "sort_kind", "python_hash_seed", "thread_env"}


class TestPaths:
    def test_project_root_contains_the_configs_directory(self) -> None:
        assert (paths.project_root() / "configs" / "base.yaml").exists()

    def test_resolve_is_independent_of_the_working_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The property that makes runs machine-independent."""
        before = paths.resolve("models")
        monkeypatch.chdir(tmp_path)
        assert paths.resolve("models") == before

    def test_absolute_inputs_pass_through(self, tmp_path: Path) -> None:
        assert paths.resolve(str(tmp_path)) == tmp_path

    def test_root_override_is_honoured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("ML_PLATFORM_ROOT", str(tmp_path))
        assert paths.project_root() == tmp_path.resolve()

    def test_relative_to_root_strips_machine_specific_prefixes(self) -> None:
        """Run records must never carry an absolute local path."""
        rendered = paths.relative_to_root(paths.project_root() / "models" / "m.joblib")
        assert rendered == "models/m.joblib"
        assert ":" not in rendered

    def test_ensure_dir_creates_and_returns(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b"
        assert paths.ensure_dir(target) == target
        assert target.is_dir()
