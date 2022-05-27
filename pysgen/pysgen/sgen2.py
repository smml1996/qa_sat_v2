import numpy as np
from pysgen.constraints import writeConstraint


def exactlyOne(vars):
    yield "ONEOF( " + ", ".join(map(str, vars)) + " );"


def doconstraints(cubesize):
    nvars = cubesize*cubesize*cubesize
    varmap = dict()
    matr = np.matrix([[[i * cubesize * cubesize + j * cubesize + k for k in range(cubesize)]
                                                                    for j in range(cubesize)]
                                                                    for i in range(cubesize)])

    for i in range(cubesize):
        for j in range(cubesize):
            yield exactlyOne(list(matr[i,j,:]))
            yield exactlyOne(list(matr[:, i, j]))
            yield exactlyOne(list(matr[i, :, j]))

def main(n):

    for group in doconstraints(n):
        print group
