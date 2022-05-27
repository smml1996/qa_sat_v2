import sys
from copy import deepcopy

sys.path.append("../")
from utils import *


args = sys.argv
file = args[1]

num_variables, num_clauses, variables, clauses = load_cnf(f"./{file}.cnf")

count_long_clauses = 0
long_clauses = []

for clause in clauses:
    if len(clause) > 2:
        count_long_clauses += 1
        long_clauses.append(clause)


for i in range(1,count_long_clauses+1):
    new_file_name = f"{file}_xor{i}.cnf"

    current_clauses = deepcopy(clauses)
    for clause in long_clauses[:i]:
        _, new_clauses, _ = xor_clause(num_variables, clause)
        current_clauses.extend(new_clauses)
    dump_clauses_to_cnf_file(f"./{file}/{new_file_name}", current_clauses, num_variables, [f"xor clauses: {i}"])







