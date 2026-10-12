"""Cargo invocation context preserving repository and stack configuration."""

from __future__ import annotations

import tempfile
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from benchlib.budget import ConfigurationError

OVERLAY_BEGIN = "# >>> atlas stack development overlay (generated) >>>"


@dataclass(frozen=True)
class CargoContext:
    """Working directory and command prefix for one coherent Cargo context."""

    working_directory: Path
    command_prefix: tuple[str, ...]


def _toolchain_channel(repository: Path) -> str:
    toolchain = repository / "rust-toolchain.toml"
    try:
        with toolchain.open("rb") as stream:
            document = tomllib.load(stream)
        channel = document["toolchain"]["channel"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(
            f"cannot read pinned toolchain from {toolchain}: {error}"
        ) from error
    if not isinstance(channel, str) or not channel:
        raise ConfigurationError(f"toolchain.channel in {toolchain} must be a string")
    return channel


def _enclosing_stack_configuration(repository: Path) -> Path | None:
    for directory in [repository, *repository.parents]:
        candidate = directory / ".cargo" / "config.toml"
        if candidate.is_file() and OVERLAY_BEGIN in candidate.read_text(encoding="utf-8"):
            return candidate
    return None


def overlay_free_configuration(configuration: Path) -> str:
    """Preserve stack build settings while excluding development patches."""
    source = configuration.read_text(encoding="utf-8")
    before_overlay, marker, _ = source.partition(OVERLAY_BEGIN)
    if not marker:
        raise ConfigurationError(f"stack overlay marker is absent from {configuration}")
    lines = before_overlay.rstrip().splitlines()
    target_line = next(
        (
            index
            for index, line in enumerate(lines)
            if line.split("#", 1)[0].strip().startswith("target-dir")
        ),
        None,
    )
    if target_line is None:
        raise ConfigurationError(f"stack target-dir is absent from {configuration}")
    _, _, raw_value = lines[target_line].partition("=")
    value = Path(raw_value.strip().strip('"').strip("'"))
    target = value if value.is_absolute() else configuration.parent.parent / value
    lines[target_line] = f'target-dir = "{target.resolve().as_posix()}"'
    return "\n".join(lines) + "\n"


@contextmanager
def cargo_context(repository: Path) -> Iterator[CargoContext]:
    """Yield Cargo settings with the stack source overlay excluded."""
    channel = _toolchain_channel(repository)
    prefix = ["rustup", "run", channel, "cargo"]
    stack_configuration = _enclosing_stack_configuration(repository)
    if stack_configuration is None:
        yield CargoContext(repository, tuple(prefix))
        return

    member_configuration = repository / ".cargo" / "config.toml"
    if member_configuration.is_file():
        prefix.extend(["--config", str(member_configuration)])
    with tempfile.TemporaryDirectory(prefix="leto-bench-") as temporary:
        working_directory = Path(temporary)
        cargo_directory = working_directory / ".cargo"
        cargo_directory.mkdir()
        (cargo_directory / "config.toml").write_text(
            overlay_free_configuration(stack_configuration), encoding="utf-8"
        )
        yield CargoContext(working_directory, tuple(prefix))
