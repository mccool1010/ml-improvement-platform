"""Tests for the command line surface.

The CLI is the documented way to reproduce a result, so its contract matters:
commands must exist, arguments must parse, exit codes must be meaningful, and a
failure must be non-zero so continuous integration can gate on it.

One structural property is tested explicitly. Native thread pools read their
environment when the library first loads, so the CLI must pin threads before
numpy or scikit-learn are imported. If an import ever moves to module scope the
pinning silently stops working, and nothing else would notice.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ml_platform import cli
from ml_platform.paths import project_root

HEAVY_IMPORTS = {"numpy", "pandas", "sklearn", "scipy", "joblib", "pandera"}
EXPECTED_COMMANDS = {"download", "validate", "train", "reproduce"}


class TestParser:
    def test_every_command_is_registered(self) -> None:
        parser = cli.build_parser()
        actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
        subcommands = next((set(a.choices) for a in actions if "train" in (a.choices or {})), set())
        assert subcommands >= EXPECTED_COMMANDS

    def test_a_command_is_required(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([])

    def test_an_unknown_command_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["deploy"])

    def test_production_is_the_default_environment(self) -> None:
        """Development subsamples the data, so it must never be the default."""
        assert cli.build_parser().parse_args(["train"]).environment == "production"

    def test_an_unknown_environment_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["--environment", "staging", "train"])

    def test_baseline_is_the_default_model(self) -> None:
        assert cli.build_parser().parse_args(["train"]).model == "baseline"

    def test_an_unknown_model_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["train", "--model", "xgboost"])

    def test_train_flags_parse(self) -> None:
        args = cli.build_parser().parse_args(
            ["train", "--model", "candidate", "--no-save", "--nrows", "500"]
        )
        assert args.model == "candidate"
        assert args.no_save is True
        assert args.nrows == 500

    def test_strict_is_the_default_tolerance_profile(self) -> None:
        """The looser profile must be opt-in, never the silent default."""
        assert cli.build_parser().parse_args(["reproduce"]).profile == "strict"

    def test_both_tolerance_profiles_are_accepted(self) -> None:
        for profile in ("strict", "portable"):
            args = cli.build_parser().parse_args(["reproduce", "--profile", profile])
            assert args.profile == profile

    def test_an_unknown_profile_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["reproduce", "--profile", "lenient"])

    def test_every_command_binds_a_handler(self) -> None:
        parser = cli.build_parser()
        for command in sorted(EXPECTED_COMMANDS):
            assert callable(parser.parse_args([command]).handler), command


class TestHelp:
    @pytest.mark.parametrize("argv", [["--help"], ["train", "--help"], ["reproduce", "--help"]])
    def test_help_exits_cleanly(self, argv: list[str], capsys: pytest.CaptureFixture) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli.main(argv)
        assert excinfo.value.code == 0
        assert capsys.readouterr().out


@pytest.fixture(scope="module")
def cli_module() -> ast.Module:
    """The CLI source, parsed, so its import order can be inspected."""
    source = (project_root() / "src" / "ml_platform" / "cli.py").read_text(encoding="utf-8")
    return ast.parse(source)


class TestThreadPinningHappensBeforeHeavyImports:
    """Structural guard on the CLI's deliberate import order."""

    def test_no_heavy_library_is_imported_at_module_scope(self, cli_module: ast.Module) -> None:
        """A module-scope numpy import would defeat thread pinning entirely."""
        offenders = []
        for node in cli_module.body:
            if isinstance(node, ast.Import):
                offenders += [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                offenders.append(node.module.split(".")[0])
        assert not (set(offenders) & HEAVY_IMPORTS)

    def test_the_bootstrap_pins_threads(self) -> None:
        from ml_platform.determinism import THREAD_ENV_VARS

        config = cli._bootstrap("production")
        for variable in THREAD_ENV_VARS:
            import os

            assert os.environ[variable] == str(config.n_threads)


class TestValidateCommand:
    def test_it_reports_splits_as_json_and_exits_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        synthetic_config: object,
        capsys: pytest.CaptureFixture,
    ) -> None:
        import json

        monkeypatch.setattr(cli, "load_config", lambda _environment: synthetic_config)
        exit_code = cli.main(["validate"])
        assert exit_code == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["observable_rows"] > 0
        assert {s["name"] for s in payload["splits"]} == {
            "train",
            "validation",
            "test",
            "production_stream",
        }

    def test_a_validation_failure_exits_non_zero(
        self, monkeypatch: pytest.MonkeyPatch, synthetic_config: object
    ) -> None:
        """Continuous integration gates on the exit code, so it must be honest."""
        from ml_platform.data.validation import DataValidationError

        def _fail(_config: object) -> None:
            raise DataValidationError("schema failed")

        monkeypatch.setattr(cli, "load_config", lambda _environment: synthetic_config)
        monkeypatch.setattr("ml_platform.pipelines.train_pipeline.load_prepared_dataset", _fail)
        assert cli.main(["validate"]) == 1


class TestTrainCommand:
    def test_it_trains_and_exits_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        synthetic_config: object,
        capsys: pytest.CaptureFixture,
    ) -> None:
        monkeypatch.setattr(cli, "load_config", lambda _environment: synthetic_config)
        monkeypatch.setattr(
            "ml_platform.pipelines.train_pipeline.load_config",
            lambda _environment: synthetic_config,
        )
        assert cli.main(["train", "--no-save"]) == 0
        assert "test AP=" in capsys.readouterr().out


class TestScriptWrappers:
    """The scripts in scripts/ must stay thin wrappers, not a second code path."""

    @pytest.mark.parametrize(
        "script", ["train.py", "download_data.py", "validate_data.py", "reproduce.py"]
    )
    def test_each_wrapper_delegates_to_the_cli(self, script: str) -> None:
        source = (project_root() / "scripts" / script).read_text(encoding="utf-8")
        assert "from ml_platform.cli import main" in source

    @pytest.mark.parametrize(
        "script", ["train.py", "download_data.py", "validate_data.py", "reproduce.py"]
    )
    def test_no_wrapper_manipulates_the_import_path(self, script: str) -> None:
        """sys.path surgery would mean the installed package is not what runs."""
        source = (project_root() / "scripts" / script).read_text(encoding="utf-8")
        assert "sys.path" not in source

    def test_the_package_exposes_a_module_entry_point(self) -> None:
        assert (Path(project_root()) / "src" / "ml_platform" / "__main__.py").exists()
