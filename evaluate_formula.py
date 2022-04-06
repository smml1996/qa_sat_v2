import sys
from typing import Dict
from utils import evaluate_clause

# arg_1 = folder (sat, unsat)
# arg_2 = num_variables [140,300]
# arg3 = path with variable's value to evaluate in cnf format
folder = sys.argv[1]
_variables = sys.argv[2]
path = f"./{folder}/sgen1-{folder}-{_variables}-100.cnf"
count_false_clauses = 0
file = open(path)
answer_path = sys.argv[3]
answer_file = open(answer_path)
answer: Dict[int, int] = dict()

cnf_result = answer_file.readline().split(" ")
assert (int(cnf_result[-1]) == 0)

for x in cnf_result[:-1]:
    int_x = int(x)
    answer[abs(int_x)] = int(int_x > 0)
answer_file.close()

num_clauses = -1
num_variables = -1
sat = True
for line in file.readlines():
    line = line[:-1]
    elements = line.split(" ")
    if elements[0] == "c":
        # this is a comment
        continue
    elif elements[0] == "p":
        assert (len(elements) == 4)
        assert (elements[1] == "cnf")
        num_variables = elements[2]
        num_clauses = elements[3]
    else:
        elements = [int(x) for x in elements]
        assert (elements[-1] == 0)
        clause = evaluate_clause(answer, elements[:-1])
        #     if -c in answer.keys():
        #         print(c)
        #         assert(answer[-c] != answer[c])
        if not clause:
            print(elements[:-1])
            count_false_clauses += 1
        sat = sat and clause
file.close()

print(f"sgen1-sat-{_variables}-100.cnf", sat, ", #vars: ", len(answer.keys()), ", false clauses:", count_false_clauses)



