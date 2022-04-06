import sys
from typing import Dict

from utils import load_cnf, cnf_to_bqm, evaluate_cnf_formula

# arg_1 = folder (sat, unsat)
# arg_2 = num_variables [140,300]
# arg3 = path with variable's value to evaluate in cnf format

# set path of CNF file
folder = sys.argv[1]
_variables = sys.argv[2]
answer_path = sys.argv[3]

path = f"./{folder}/sgen1-{folder}-{_variables}-100.cnf"

# build bqm
num_variables, num_clauses, variables, clauses = load_cnf(path)
assert(num_variables == len(variables))

file = open(answer_path)
elements_cnf_result = file.readline().split(" ")
file.close()
assert(int(elements_cnf_result[-1]) == 0)
assert(len(elements_cnf_result) == num_variables + 1)

values: Dict[int, int] = dict()
for x in elements_cnf_result[:-1]:
    int_x = int(x)
    values[abs(int_x)] = int(int_x > 0)

bqm, or_result_vars, clauses_qubits = cnf_to_bqm(variables, clauses)

# add_clauses_constraints(bqm, clauses_qubits, 3)


print(evaluate_cnf_formula(values, or_result_vars, bqm))