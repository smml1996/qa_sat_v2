from itertools import izip_longest
from random import random, sample
from random import shuffle

from pysgen.constraints import AtLeastOne, AtMostOne, TwoInFour





class Partition(object):
    def __init__(self, part, gsize):
        self.elems = part
        self.gsize = gsize
        self.map = mapof(gsize, part)

    @classmethod
    def copy(clazz, orig):
        return clazz(list(orig.elems), orig.gsize)


    def swap(self, swap1,swap2):
        newpart = self.elems
        newpart[swap1], newpart[swap2] = newpart[swap1], newpart[swap2]
        group1 = self.elems[swap1 % self.gsize:(swap1 % self.gsize)+self.gsize]
        group2 = self.elems[swap2 % self.gsize:(swap2 % self.gsize)+self.gsize]
        for elem in group1:
            self.map[elem] = set(group1)
            self.map[elem].remove(elem)

        for elem in group2:
            self.map[elem] = set(group2)
            self.map[elem].remove(elem)

    def groups(self):
        return grouper(self.gsize, self.elems)



def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

def neighs(map1,map2, elem):
    ret = set()
    for nn in map1[elem]:
        ret.update(map2[nn])
    return ret



class SgenProblem(object):
    def __init__(self, parts, groupsize, satsize):
        self.partitions = parts
        self.groupsize = groupsize
        self.satsize = satsize

    def scoresimilar(self, part1, part2):
        groupsize = self.groupsize
        score = 0

        for elem in part1.elems:
            dist = set()
            for next in part1.map[elem]:
                dist.update(part2.map[next])
            if elem in dist:
                score += 16
            dist2 = set()
            for next1 in dist:
                for next2 in part1.map[next1]:
                    dist2.update(part2.map[next2])
            if elem in dist2:
                score += 1

        return score

    def deltaScore(self, part1, part2, swap1, swap2):
        map1 = part1.map
        map2 = part2.map

        neighs1 = neighs(map1, map2, swap1)
        neighs2 = neighs(map1, map2, swap2)
        ret = 0
        # xxx fix
        if swap1 in neighs1:
            ret -= 16
        if swap2 in neighs2:
            ret -= 16

        if swap1 in neighs2:
            ret += 16
        if swap2 in neighs1:
            ret += 16

        return ret

    def anneal(self, part1, part2, steps=10000):
        ones, zeros = self.split_sat(part1)
        sat_ratio = float(self.satsize) / self.groupsize

        for i in range(steps):
            temp = 0.5 ** (i)
            if random() < sat_ratio:
                samplefrom = ones
            else:
                samplefrom = zeros
            swap1, swap2 = sample(samplefrom, 2)

            if (self.deltaScore(part1, part2, swap1, swap2)) < 0 or random() < temp:
                index1 = part1.elems.index(swap1)
                index2 = part1.elems.index(swap2)
                part1.swap(index1, index2)

    def satshuffle(self, part):
        ones, zeros = self.split_sat(part)
        del part.elems[:]

        shuffle(ones)
        shuffle(zeros)
        for head, rest in zip(grouper(self.satsize, ones), grouper(self.groupsize - self.satsize, zeros)):
            part.elems.extend(head)
            part.elems.extend(rest)
        assert len(part.elems) == len(ones) + len(zeros)
        part.map = mapof(self.groupsize, part.elems)

    def split_sat(self, part):
        ones, zeros = [], []
        for group in part.groups():
            ones.extend(group[:self.satsize])
            zeros.extend(group[self.satsize:])
        return ones, zeros

    def createProblem(self, n):
        groupsize = self.groupsize
        assert n % groupsize == 0, (n, groupsize)
        part1 = range(n)
        shuffle(part1)
        part1 = Partition(part1, self.groupsize)
        solution = []
        for group in part1.groups():
            solution.extend(group[:self.satsize])
            solution.extend(-x for x in group[self.satsize:])
        solution.sort(key=abs)
        self.ensurePartition(part1, solution)
        part2 = Partition.copy(part1)
        part3 = Partition.copy(part1)
        minscore = self.scoresimilar(part1, part2)
        # xxx  do annealing
        for _ in xrange(100):
            # print minscore
            temppart = Partition.copy(part1)
            self.satshuffle(temppart)
            self.anneal(temppart, part1)
            score = self.scoresimilar(part1, temppart)
            if score < minscore:
                part2 = temppart
                minscore = score
        minscore = self.scoresimilar(part1, part3) + self.scoresimilar(part2, part3)
        for _ in xrange(100):
            # print minscore
            temppart = Partition.copy(part1)
            self.satshuffle(temppart)
            self.anneal(temppart, part1)
            self.anneal(temppart, part2)
            score = self.scoresimilar(part1, temppart) + self.scoresimilar(part2, temppart)
            if score < minscore:
                part3 = temppart
                minscore = score
        firstConstraint, secondConstraint = self.partitions[:2]


        problem = list()

        def constrainGroups(constr, part):
            problem.extend(map(constr, part.groups()))

        constrainGroups(firstConstraint, part1)
        constrainGroups(secondConstraint, part2)
        constrainGroups(secondConstraint, part3)
        self.ensurePartition(part1, solution)
        self.ensurePartition(part2, solution)
        self.ensurePartition(part3, solution)

        return problem, solution

    def ensurePartition(self, part1, solution):
        for group in part1.groups():
            assert all(x in solution for x in group[:self.satsize])
            assert all(-x in solution for x in group[self.satsize:])


def mapof(groupsize, part1):
    map1 = dict()
    for group in grouper(groupsize, part1):
        for elem in group:
            map1[elem] = set(group)
            map1[elem].remove(elem)
    return map1

problemtypes = {
    "base": SgenProblem([AtMostOne, AtLeastOne, AtLeastOne], 5, 1),
    "opp": SgenProblem([AtLeastOne, AtMostOne, AtMostOne], 5, 1),
    "twoinfour": SgenProblem([TwoInFour, TwoInFour, TwoInFour], 4, 2)
}


def main(n, outtypes, probtype, groupsize=5):

    sgenp = problemtypes[probtype]
    sgenp.groupsize = groupsize
    assert n% groupsize == 0, (n, groupsize)
    problem, solution = sgenp.createProblem(n)

    for outtype,outfile in outtypes:
        if outtype == "cnf":
            print >>outfile, "c solution: ", " ".join(str(x + (1 if x > 0 else -1)) for x in solution)
            print >>outfile, "p cnf", n, sum(len(constr.to_cnf()) for constr in problem)
            printer = lambda  x: "\n".join(" ".join(map(str,xx)) + " 0" for xx in x.to_cnf())
        elif outtype == "bench":
            print >>outfile, "# solution: ", " ".join(map(str, solution))
            printer = lambda  x: str(x.to_bench())
        else:
            raise RuntimeError("invalid outtype: "+outtype)
        for constr in problem:
            print >>outfile, printer(constr)





