# PyHeatrs

This repository is an educational tool to show how to integrate Rust within a
Python project. The project implements the heat equation in both languages and
shows the performance advantage of Rust. With tools such as
[`maturin`](https://www.maturin.rs/) integrating Rust and Python is a breeze.
This illustrates the advantage of Rust over other low-level languages that one
could integrate with Python.

## Getting started

To get started one need to install Python and Rust. See your systems guides for
installing Python and for Rust we recommend [`rustup`](https://rustup.rs/).
Once those are installed it is recommended to create a virtual environment
within this folder to contain development and install dependencies without
worry.

> [!NOTE]
> The following is automated with
> [`nox`](https://nox.thea.codes/en/stable/index.html) and the session `dev`.
> We will use `nox` later when testing so to quickly get started install `nox`
> for your user with `python3 -m pip install --user nox` and then run
> `nox -s dev` which will automate the setup of a virtual environment for you
> and install all required development dependencies. Remember to source the
> environment after running `nox` - `source .venv/bin/activate`.

### Manual setup

Create a virtual environment with your default Python interpreter.

```bash
# Create the environment
python3 -m venv .venv --upgrade deps
# Activate the environment so that future installs only happen within the environment
source .venv/bin/activate
```

Next we need to install [`maturin`](https://www.maturin.rs/). `maturin` is a
build tool which integrates Rust into Python, it can take control of our
project and ensure that both Python and Rust code is built and can interact.

```bash
python3 -m pip install maturin
```

Then we can tell `maturin` to develop our package which will download
dependencies and build the Rust code for us.

```bash
maturin develop
```

You should now be able to try the command line tool within this project which
is intended as the access point for interactive interactions with the library.

```bash
pyheatrs --help
```

## Next steps

The heat equation is already implemented in Python and you should give it a
quick peek as well as play around with the `pyheatrs` command line tool. Once
you feel ready, head over to the `src/lib.rs` Rust file and implement the heat
equations in Rust. We recommend that you start simple and once it is working
try to improve performance.

## Testing and benchmarking

> [!IMPORTANT]
> When benchmarking it is essential to run `maturin develop --release` so that
> the Rust code is compiled with all available optimizations. This is handled
> automatically with `nox -s bench`.

To ensure that your Rust implementation works as expected we have included
tests in the `tests/` directory as well as
[`nox`](https://nox.thea.codes/en/stable/index.html) sessions (`test` and
`bench`) which you can use to quickly get check that everything works as
expected. Run `nox -s test` to check your implementation and run `nox -s bench`
to benchmark your implementation.

## Heat equations

The inspiration for this project is the [Heat equation mini-app found in
multiple ENCCS
workshops](https://enccs.github.io/sycl-workshop/heat-equation/). One can read
more about the details of the math behind the evolution of the heat equations
there.

To summaries we model the rate of change in temperature field over both the
`x`, `y` and time dimension. To manage this within the computer we discretize
over all three dimensions. This gives us a stencil operation which we can apply
to all cells of our field.

## Project structure

The project is structured as follows:
- `pyheatrs/` contains the Python code for the project
    - `main.py` contains the command line tool
    - `heat.py` contains the heat equation implementation in Python
- `src/` contains the Rust code
    - `lib.rs` contains the Rust implementation of the heat equation
- `presentation/` contains the introduction (in Norwegian) to the project
- `pyproject.toml` is the Python project configuration
    - Add new Python dependencies here if needed
- `Cargo.toml` is the Rust project configuration
    - Add new Rust dependencies here if needed
- `noxfile.py` contains the `nox` sessions defined for this project

## Visualization of output

![heat equations shown over a 2D grid](./presentation/images/result.gif)
