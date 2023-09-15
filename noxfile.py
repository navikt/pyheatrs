# /usr/bin/env python3

"""
Nox files are an easy way to automate testing in different virtual environments. It
will automatically create virtual environments and install required packages.
"""

import nox
import os
import pathlib


# Default sessions to run when just `nox` is run in the project
nox.options.sessions = ["test"]

# Path to the development virtual environment
VENV_DIR = pathlib.Path("./.venv").resolve()


@nox.session
def dev(session: nox.Session):
    """
    Sets up a python development environment for the project.

    This session will:
    - Create a python virtualenv for the session
    - Install the `virtualenv` cli tool into this environment
    - Use `virtualenv` to create a global project virtual environment
    - Invoke the python interpreter from the global project environment to install
      the project and all it's development dependencies.
    """
    if VENV_DIR.exists():
        session.error(
            f"virtualenv ({VENV_DIR}) already exists, "
            "delete it to run this session successfully"
        )
    session.install("virtualenv")
    session.run("virtualenv", os.fsdecode(VENV_DIR), silent=True)
    python = os.fsdecode(VENV_DIR / "bin/python3")
    session.run(python, "-m", "pip", "install", "maturin", external=True)
    session.run(
        python, "-m", "pip", "install", "-rrequirements-lint.txt", external=True
    )
    session.run(python, "-m", "pip", "install", "-rrequirements-dev.txt", external=True)
    session.run(python, "-m", "maturin", "develop", external=True)


@nox.session
def lint(session: nox.Session):
    session.install("-rrequirements-lint.txt")
    session.run("black", ".")
    session.run("ruff", "check", ".")


@nox.session
def test(session: nox.Session):
    session.install("maturin")
    session.install("-rrequirements-dev.txt")
    session.run_always("maturin", "develop")
    session.run("pytest", "--benchmark-disable")


@nox.session
def bench(session: nox.Session):
    session.install("maturin")
    session.install("-rrequirements-dev.txt")
    session.install(".")
    session.run("pytest", "--benchmark-only")
