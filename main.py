# -*- coding: utf-8 -*-
"""Command-line entry point for the MLforCO experiment pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PHASES = {
    "generate": "01_generate_instances.py",
    "solve": "02_solve_optimal.py",
    "train": "03_train_model.py",
    "evaluate": "04_evaluate_matheuristic.py",
}

PHASE_ORDER = ("generate", "solve", "train", "evaluate")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Punto de entrada oficial para ejecutar el pipeline "
            "experimental ML-assisted CO sobre MCLSP."
        )
    )
    parser.add_argument(
        "command",
        choices=("all", *PHASES.keys()),
        help=(
            "Fase a ejecutar: all, generate, solve, train o evaluate."
        ),
    )
    return parser


def resolve_scripts(command: str) -> list[str]:
    """Return the script sequence associated with a CLI command."""
    if command == "all":
        return [PHASES[phase] for phase in PHASE_ORDER]
    return [PHASES[command]]


def print_phase_header(script_name: str) -> None:
    """Print a clear visual separator before running a phase."""
    separator = "=" * 60
    print(separator)
    print(f"Ejecutando: {script_name}")
    print(separator)


def run_script(script_name: str, project_root: Path) -> int:
    """Run one pipeline script and return its exit code."""
    script_path = project_root / script_name

    if not script_path.is_file():
        print(
            f"Error: no se encontró el archivo requerido: {script_name}",
            file=sys.stderr,
        )
        return 1

    print_phase_header(script_name)

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root),
            check=False,
        )
    except OSError as exc:
        print(
            f"Error ejecutando {script_name}: {exc}",
            file=sys.stderr,
        )
        return 1

    if completed.returncode != 0:
        print(
            (
                f"Error: la fase {script_name} falló con código "
                f"{completed.returncode}."
            ),
            file=sys.stderr,
        )
        return completed.returncode

    return 0


def run_pipeline(command: str, project_root: Path) -> int:
    """Run the requested pipeline command."""
    for script_name in resolve_scripts(command):
        exit_code = run_script(script_name, project_root)
        if exit_code != 0:
            return exit_code

    print("Experimento completado exitosamente.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and execute the requested phase."""
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parent
    return run_pipeline(args.command, project_root)


if __name__ == "__main__":
    sys.exit(main())
