#!/usr/bin/env python3

import argparse
import sys

import numpy as np

from wolf_sheep_grass import WolfSheepGrassModel


def main():
    # get command line arguments
    parser = argparse.ArgumentParser(description="Wolf sheep grass simulator")

    parser.add_argument(
        "-o", "--output_file", help="output for csv data", type=str, default="-"
    )
    parser.add_argument(
        "-mt",
        "--maximum_time",
        help="maximum time to run the simulator",
        type=float,
        default=float("inf"),
    )
    parser.add_argument("-gw", "--grid_width", type=int, help="grid width", default=51)
    parser.add_argument(
        "-gh", "--grid_height", type=int, help="grid height", default=51
    )
    parser.add_argument(
        "-iw", "--init_wolves", type=int, help="init number of wolves", default=50
    )
    parser.add_argument(
        "-is", "--init_sheep", type=int, help="init number of sheep", default=100
    )
    parser.add_argument(
        "-wgf",
        "--wolf_gain_from_food",
        type=float,
        help="wolf gain from food",
        default=20.0,
    )
    parser.add_argument(
        "-sgf",
        "--sheep_gain_from_food",
        type=float,
        help="sheep gain from food",
        default=4.0,
    )
    parser.add_argument(
        "-wrep", "--wolf_reproduce", type=float, help="wolf reproduce", default=5.0
    )
    parser.add_argument(
        "-srep", "--sheep_reproduce", type=float, help="sheep reproduce", default=4.0
    )
    parser.add_argument(
        "-igp",
        "--init_grass_proportion",
        type=float,
        help="initial grass proportion (approx)",
        default=0.5,
    )
    parser.add_argument(
        "-grt",
        "--grass_regrowth_time",
        type=float,
        help="grass regrowth time",
        default=30.0,
    )

    args = parser.parse_args()

    # create model
    model = WolfSheepGrassModel(
        GRID_WIDTH=args.grid_width,
        GRID_HEIGHT=args.grid_height,
        INIT_WOLVES=args.init_wolves,
        WOLF_GAIN_FROM_FOOD=args.wolf_gain_from_food,
        WOLF_REPRODUCE=args.wolf_reproduce,
        INIT_SHEEP=args.init_sheep,
        SHEEP_GAIN_FROM_FOOD=args.sheep_gain_from_food,
        SHEEP_REPRODUCE=args.sheep_reproduce,
        INIT_GRASS_PROPORTION=args.init_grass_proportion,
        GRASS_REGROWTH_TIME=args.grass_regrowth_time,
    )

    # find the output source
    if args.output_file == "-":
        file = sys.stdout
    else:
        file = open(args.output_file, "w")

    # run the simulator
    try:
        tick = 0
        print(f"time,number of wolves, number of sheep, number of grass", file=file)
        print(
            f"{tick},{model.num_wolves},{model.num_sheep},{np.sum(model.grass)}",
            file=file,
        )

        while (
            tick < args.maximum_time
            and model.num_wolves > 0
            and model.num_sheep < model.MAX_SHEEP
        ):
            # self.sheep
            model.sheep_move()
            model.sheep_energy -= 1.0  # self.sheep metabolism
            model.sheep_eat_grass()
            model.sheep_die()
            model.sheep_reproduce()

            # wolves
            model.wolves_move()
            model.wolf_energy -= 1.0  # wolf metabolism
            model.wolves_eat_sheep()
            model.wolves_die()
            model.wolves_reproduce()

            # grass
            model.grow_grass()

            tick += 1

            if tick % 100 == 0:
                model.compact_sheep_arrays()
                model.compact_wolf_arrays()

            print(
                f"{tick},{model.num_wolves},{model.num_sheep},{np.sum(model.grass)}",
                file=file,
            )
    finally:
        file.close()


if __name__ == "__main__":
    # execute only if run as a script
    main()
