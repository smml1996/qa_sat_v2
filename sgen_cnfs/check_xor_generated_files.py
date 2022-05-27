import sys

sys.path.append("../")

from utils import *
from brute_force import *

files = {
    10: 4,
    20: 8
}


for (cvars, cxors) in files.items():
    original_cvars, _, original_variables, original_clauses = load_cnf(f"./var{cvars}.cnf")
    count_answers, answers = get_count_satisfiable(original_variables, original_clauses, return_answers=True)

    for cxor in range(1,cxors+1):
        num_variables, num_clauses, variables, clauses = load_cnf(f"./var{cvars}/var{cvars}_xor{cxor}.cnf")
        count_xor_file = get_count_satisfiable(variables, clauses)

        assert(count_xor_file == count_answers)
        for answer in answers:
            assert(evaluate_clauses(answer, clauses))
