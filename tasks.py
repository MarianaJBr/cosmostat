"""
Collection of development tasks.

Usage:
    python -m tasks TASK-NAME
"""

from subprocess import run
from typing import List

import click

from cosmostat import __version__

# Arguments to pass to subprocess.run for each task.
FORMAT_ARGS = ["black", "tasks.py", "src", "tests"]
ISORT_ARGS = ["isort", "tasks.py", "src", "tests"]
PYDOCSTYLE_ARGS = ["pydocstyle", "tasks.py", "src", "tests"]
PYTEST_ARGS = ["pytest"]
MYPY_ARGS = ["mypy", "src", "tests"]


def _run(command: List[str]):
    """Run a subcommand through python subprocess.run routine."""
    # NOTE: See https://stackoverflow.com/a/32799942 in case we want to
    #  remove shell=True.
    run(command)


@click.group("tasks")
def app():
    """Main entry point."""
    pass


@app.command()
def black():
    """Format the source code using black."""
    _run(FORMAT_ARGS)


@app.command()
def isort():
    """Format imports using isort."""
    _run(ISORT_ARGS)


@app.command()
def pydocstyle():
    """Run pydocstyle."""
    _run(PYDOCSTYLE_ARGS)


@app.command()
def mypy():
    """Run mypy."""
    _run(MYPY_ARGS)


@app.command()
def tests():
    """Run test suite."""
    _run(PYTEST_ARGS)


@app.command()
def version():
    """Run mypy."""
    print(__version__)


@app.command(name="format")
def format_():
    """Run all formatting tasks."""
    _run(FORMAT_ARGS)
    _run(ISORT_ARGS)


@app.command()
def format_check():
    """Run all formatting and typechecking tasks."""
    _run(FORMAT_ARGS)
    _run(ISORT_ARGS)
    _run(MYPY_ARGS)


if __name__ == "__main__":
    app(prog_name="tasks")
