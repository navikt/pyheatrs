#!/usr/bin/env python3

"""
Command line tool for the Rust library PyHeatrs
"""

import argparse
import numpy as np
import pyheatrs


def __parse_size(size: str) -> tuple[int, int]:
    """Helper function to parse resolution size and return `(width, height)`"""
    if size == "720":
        return (1280, 720)
    elif size == "1080":
        return (1920, 1080)
    elif size == "4k":
        return (3840, 2160)
    elif size == "8k":
        return (7680, 4320)
    else:
        # If an unknown size is given we return a default size
        return (640, 480)


def __render(args):
    """Helper function to render heat equation using `matplotlib`"""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    field_update = pyheatrs.py.evolve if args.impl == "python" else pyheatrs.rs.evolve
    field = pyheatrs.py.default_field((args.width, args.height))
    fig, ax = plt.subplots()
    img = ax.imshow(field.T, cmap="viridis", aspect="auto", interpolation="nearest")

    def _update_fig(_, field):
        field[...] = field_update(
            field,
            args.diffusion,
            args.dt,
            args.steps,
        )
        img.set_data(field.T)
        avg_tmp = np.mean(field[1:-2, 1:-2])
        ax.set_title(f"Avg temp: {avg_tmp:5.3f}")
        return (img,)

    ani = FuncAnimation(fig, _update_fig, fargs=(field,), frames=args.frames)
    if not args.file:
        plt.show()
    else:
        ani.save(args.file)


def main_cli():
    """Entry point for command line tool"""
    parser = argparse.ArgumentParser(
        prog="pyheatrs", description="Helper to render PyHeatrs"
    )
    parser.add_argument("--width", type=int, default=640, help="Width of 2D structure")
    parser.add_argument(
        "--height", type=int, default=480, help="Height of 2D structure"
    )
    parser.add_argument(
        "--size",
        choices=["720", "1080", "4k", "8k"],
        help="Use specific size for 2D structure",
    )
    parser.add_argument(
        "--diffusion",
        type=float,
        default=0.5,
        help="Diffusion constant for heat equation",
    )
    parser.add_argument("--file", help="Filename to save render")
    parser.add_argument(
        "-i",
        "--impl",
        choices=["python", "rust"],
        help="Which implementation should be used to render",
        default="python",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=1000,
        help="Number of iterations to evolve the heat equations",
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="Number of steps between renders"
    )
    args = parser.parse_args()
    if args.size:
        args.width, args.height = __parse_size(args.size)
    args.dt = pyheatrs.py.estimate_dt((args.width, args.height), args.diffusion)
    __render(args)


if __name__ == "__main__":
    main_cli()
