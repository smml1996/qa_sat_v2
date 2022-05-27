# Pysgen: python implementation of  the sgen algorithm #
This script generates satisfiable sgen instances. uses python2/pypy.
## Usage:

```
#!bash

./main.py --examples2 # generates 10 examples for various sizes
./main.py -s PROBLEM_SIZE -c output.cnf -b output.bench [-g GROUP_SIZE] [-t base|twoinfour] # generate a single problem
```