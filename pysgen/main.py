#!/usr/bin/env pypy
from pysgen.sgen import main
from sys import argv
import sys
import argparse
import random

args = argparse.ArgumentParser()

args.add_argument('-c', '--output-cnf', action="append", type=argparse.FileType("w"))
args.add_argument('-b', '--output-bench', action="append", type=argparse.FileType("w"))
args.add_argument('-s', '--size', type=int)
args.add_argument('-g', '--group-size', type=int, default=5)
args.add_argument('-t', '--type', default="base")
args.add_argument('-e', '--examples', action='store_true', default=False)
args.add_argument( '--examples2', action='store_true', default=False)
args.add_argument( '--allbenchmarks', action='store_true', default=False)


opts =  args.parse_args()


if opts.examples:
    for size in [60,80,100,120,140,160]:
        # for opp in {True, False}:
        #     with open("examples/example-{}-{}.{}".format(size,opp, "bench"), "w") as f:
        #         with open("examples/example-{}-{}.{}".format(size,opp, "cnf"), "w") as f2:
        #for gsize in {4,5,6}:
        #    if size % gsize != 0:
        #        size_ = size +  gsize - (size % gsize) # increase size to make it divisible by group size
        #    else:
        #        size_ = size
        size_ = size
        gsize = 4
        with open("examples/example-{}-g{}.{}".format(size_, gsize, "bench"), "w") as f:
            with open("examples/example-{}-g{}.{}".format(size_, gsize, "cnf"), "w") as f2:
                main(size_, [("bench", f), ("cnf", f2)], "twoinfour", gsize)
elif opts.examples2:
    for size in range(32, 160,4):
      for ptype, gsize in {("base",4), ("base",5), ("base", 6), ("twoinfour", 4)}:
        if size % gsize:
            continue
        for seed in range(10):
          random.seed(seed)
          fname = "benchmarks/sgen-{}-s{}-g{}-{}".format(ptype, size, gsize, seed)
          with open(fname + ".bench", "w") as f:
              with open(fname + ".cnf", "w") as f2:
                main(size, [("bench", f), ("cnf", f2)], ptype, gsize)
elif opts.allbenchmarks:
    for size in range(32, 201):
      for ptype, gsize in {("base",4), ("base",5), ("base", 6), ("twoinfour", 4)}:
        if size % gsize:
            continue
        for seed in range(100):
          random.seed(seed)
          fname = "allbenchmarks/sgen-{}-s{}-g{}-{}".format(ptype, size, gsize, seed)
          with open(fname + ".bench", "w") as f:
              with open(fname + ".cnf", "w") as f2:
                main(size, [("bench", f), ("cnf", f2)], ptype, gsize)
elif len(sys.argv) == 1:
    #main(60, [("bench", sys.stdout),("cnf", sys.stdout)], "base", opts.group_size)
    main(60, [("bench", sys.stdout),("cnf", sys.stdout)], "twoinfour", opts.group_size)
else:
    outs = []
    for elem in opts.output_cnf:
        outs.append(("cnf", elem))

    # for elem in opts.output_bench:
    #     outs.append((("bench", elem)))

    main(opts.size, outs, opts.type, opts.group_size)

