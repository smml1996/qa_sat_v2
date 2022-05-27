from itertools import combinations
from random import shuffle


class Constraint(object):
    def __init__(self, name, vars):
        self.name = name
        self.vars = vars

    def to_cnf(self):
        raise NotImplementedError

    def to_bench(self):
        return str(self.name)+ "(" + ", ".join(map(str, self.vars)) + ");"


class AtLeastOne(Constraint):
    def __init__(self, vars):
        super(AtLeastOne, self).__init__("ATLEASTONE", vars)

    def to_cnf(self):
        return  [tuple(x+1 for x in self.vars)]


class AtMostOne(Constraint):
    def __init__(self, vars):
        super(AtMostOne, self).__init__("ATMOSTONE", vars)

    def to_cnf(self):
        return [(-x-1,-y-1) for x,y in combinations(self.vars,2)]

class TwoInFour(Constraint):
    def __init__(self, vars):
        super(TwoInFour, self).__init__("TWOINFOUR", vars)

    def to_cnf(self):
        return [ (-a-1, -b-1, -c-1) for a,b,c in combinations(self.vars, 3)] + [ (a+1, b+1, c+1) for a,b,c in combinations(self.vars, 3)]


def atLeast1cnf(vars):
    clause = [a+ 1 for a  in vars]
    shuffle(clause)
    clause.append(0)
    yield " ".join(map(str, clause))


def atLeast1bench(vars):
    clause = list(vars)
    shuffle(clause)
    yield "ATLEASTONE( " + ", ".join(map(str, clause)) + " );"


def atMost1cnf(vars):
    clauses =  [[-a-1, -b-1] for a,b in combinations(vars, 2)]
    for clause in clauses:
        shuffle(clause)
        clause.append(0)
        yield " ".join(map(str, clause))


def atMost1bench(vars):
    clause = list(vars)
    shuffle(clause)
    yield "ATMOSTONE( " + ", ".join(map(str, clause)) + " );"


def writeConstraint(part, type):
    for clause in type(part):
        print clause


