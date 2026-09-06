"""Tests for configuration loading.

Configuration decides the label horizon, the exposure restriction, the split
boundaries and the operating point. A silent change to any of those changes every
result, so these tests cover the layering rules and the fingerprint that makes a
change visible in the run record.
"""

from __future__ import annotations

from datetime import date
from itertools import pairwise
from pathlib import Path

import pytest
import yaml

from ml_platform.config import CONFIG_LAYERS, Config, DateWindow, _deep_merge, load_config


class TestDeepMerge:
    def test_later_layers_win(self) -> None:
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_keys_merge_rather_than_replace(self) -> None:
        merged = _deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}})
        assert merged == {"a": {"x": 1, "y": 3}}

    def test_inputs_are_not_mutated(self) -> None:
        base = {"a": {"x": 1}}
        _deep_merge(base, {"a": {"x": 2}})
        assert base == {"a": {"x": 1}}

    def test_a_scalar_replaces_a_mapping(self) -> None:
        assert _deep_merge({"a": {"x": 1}}, {"a": 5}) == {"a": 5}


class TestDateWindow:
    def test_string_dates_are_parsed(self) -> None:
        window = DateWindow.from_mapping({"start": "2000-01-01", "end": "2003-12-31"})
        assert window.start == date(2000, 1, 1)
        assert window.end == date(2003, 12, 31)

    def test_reversed_window_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="after end"):
            DateWindow(start=date(2005, 1, 1), end=date(2004, 1, 1))

    def test_single_day_window_is_allowed(self) -> None:
        window = DateWindow(start=date(2004, 1, 1), end=date(2004, 1, 1))
        assert window.start == window.end


class TestLoadConfig:
    def test_production_loads_every_layer(self) -> None:
        config = load_config("production")
        assert config.environment == "production"
        assert config.raw["environment"] == "production"
        # model.yaml contributes these, base.yaml the rest.
        assert {"baseline", "candidate"} <= set(config.raw)

    def test_environment_overlay_overrides_base(self) -> None:
        development = load_config("development")
        production = load_config("production")
        assert development.sample_fraction < production.sample_fraction
        assert production.sample_fraction == 1.0

    def test_missing_environment_file_is_an_error(self) -> None:
        with pytest.raises(FileNotFoundError, match="missing configuration file"):
            load_config("staging")

    def test_layers_are_applied_in_order(self, tmp_path: Path) -> None:
        """The last layer wins, which is what makes an overlay useful."""
        for name in CONFIG_LAYERS:
            (tmp_path / name).write_text("seed: 1\n", encoding="utf-8")
        (tmp_path / "custom.yaml").write_text("seed: 99\n", encoding="utf-8")
        assert load_config("custom", tmp_path).seed == 99


@pytest.fixture(scope="module")
def config() -> Config:
    """The shipped production configuration."""
    return load_config("production")


class TestAccessors:
    def test_label_and_population_settings(self, config: Config) -> None:
        assert config.horizon_months == 60
        assert config.min_term_months == 60
        assert config.target_column == "target"

    def test_horizon_and_minimum_term_agree(self, config: Config) -> None:
        """Uniform exposure only holds while these two match.

        If the horizon changed without the minimum term following, loans would
        mature inside the observation window again and Term would start encoding
        the boundary. See docs/decisions/ADR-001.
        """
        assert config.min_term_months == config.horizon_months

    def test_observation_end_is_a_date(self, config: Config) -> None:
        assert config.observation_end == date(2014, 6, 25)

    def test_evaluation_settings(self, config: Config) -> None:
        assert config.primary_metric == "average_precision"
        assert 0 < config.review_capacity < 1

    def test_determinism_settings(self, config: Config) -> None:
        assert config.n_threads >= 1

    def test_split_names_cover_the_configured_windows(self, config: Config) -> None:
        assert set(config.split_names) == {"train", "validation", "test", "production_stream"}

    def test_windows_are_ordered_and_disjoint(self, config: Config) -> None:
        windows = sorted((config.window(n) for n in config.split_names), key=lambda w: w.start)
        for earlier, later in pairwise(windows):
            assert earlier.end < later.start

    def test_paths_resolve_absolutely(self, config: Config) -> None:
        for path in (
            config.raw_path,
            config.processed_dir,
            config.report_dir,
            config.model_dir,
            config.benchmark_dir,
        ):
            assert path.is_absolute()

    def test_paths_do_not_depend_on_the_working_directory(
        self, config: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        before = config.raw_path
        monkeypatch.chdir(tmp_path)
        assert load_config("production").raw_path == before

    def test_source_records_a_checksum(self, config: Config) -> None:
        assert len(str(config.source["sha256"])) == 64


class TestFingerprint:
    def test_identical_configuration_gives_the_same_fingerprint(self) -> None:
        assert load_config("production").fingerprint() == load_config("production").fingerprint()

    def test_different_environments_differ(self) -> None:
        assert load_config("production").fingerprint() != load_config("development").fingerprint()

    def test_any_value_change_moves_the_fingerprint(self) -> None:
        """The point of the fingerprint: a silent edit cannot hide."""
        base = load_config("production")
        altered = Config(raw={**base.raw, "seed": base.seed + 1}, environment="production")
        assert altered.fingerprint() != base.fingerprint()

    def test_key_order_does_not_affect_the_fingerprint(self) -> None:
        base = load_config("production")
        reordered = Config(raw=dict(reversed(list(base.raw.items()))), environment="production")
        assert reordered.fingerprint() == base.fingerprint()

    def test_fingerprint_is_short_and_hex(self) -> None:
        fingerprint = load_config("production").fingerprint()
        assert len(fingerprint) == 16
        assert set(fingerprint) <= set("0123456789abcdef")


class TestShippedConfigFiles:
    """The committed YAML must stay loadable and internally consistent."""

    def test_every_config_file_parses(self) -> None:
        from ml_platform.config import CONFIG_DIR

        for path in CONFIG_DIR.glob("*.yaml"):
            with path.open("r", encoding="utf-8") as handle:
                assert yaml.safe_load(handle) is not None, path.name

    def test_both_models_declare_a_known_feature_set(self) -> None:
        config = load_config("production")
        for key in ("baseline", "candidate"):
            assert config.raw[key]["feature_set"] in {"core", "engineered"}

    def test_both_models_pin_a_random_state(self) -> None:
        """An unseeded estimator would break reproduction."""
        config = load_config("production")
        for key in ("baseline", "candidate"):
            assert "random_state" in config.raw[key]["params"]
