from utils import evaluate_clauses
from cnfgen.families.randomformulas import RandomKCNF
import copy
from utils import cnf_to_bqm, evaluate_cnf_formula

def get_values_dictionary(val, variables):
    current_val = val
    answer = dict()
    for var in variables:
        answer[var] = current_val % 2
        current_val//=2
    return answer

def get_percentage_satisfiable(variables, clauses):
    current_value = 0
    answer = 0
    while current_value < 2**len(variables):
        answer_dict = get_values_dictionary(current_value, variables)
        current_value += 1
        if evaluate_clauses(answer_dict, clauses):
            answer +=1
    return round(float(answer)/float(2**len(variables)),2)

def get_count_satisfiable(variables, clauses, return_answers=False):
    current_value = 0
    answer = 0
    sat_answers = []
    while current_value < 2 ** len(variables):
        answer_dict = get_values_dictionary(current_value, variables)
        current_value += 1
        if evaluate_clauses(answer_dict, clauses):
            sat_answers.append(answer_dict)
            answer += 1
    if return_answers:
        return answer, sat_answers
    else:
        return answer



def get_random_instance(n_variables, n_clauses, seed, clause_width=3):
    cnf = RandomKCNF(clause_width, n_variables, n_clauses, seed)
    c = 1
    variables_mapping = dict()

    variables = set()
    answer = []
    for clause in cnf:
        temp_answer = []
        for (b, literal) in clause:
            if literal not in variables_mapping.keys():
                variables_mapping[literal] = c
                variables.add(c)
                c += 1
            var = variables_mapping[literal]
            if not b:
                var *=-1
            temp_answer.append(var)

        answer.append(temp_answer)
    return variables, answer


def get_bqm_random_sat(count_variables, clauses, seed):
    variables_1, clauses = get_random_instance(count_variables, clauses, seed)

    variables = copy.deepcopy(variables_1)

    bqm, or_result_vars, clauses_qubits, _ = cnf_to_bqm(variables, clauses)

    # current_value = 0
    # ratio_sat = 0.0
    # while current_value < 2 ** count_variables:
    #     answer_dict = get_values_dictionary(current_value, variables_1)
    #     assert(len(answer_dict.keys()) == count_variables)
    #     current_value += 1
    #     if evaluate_clauses(answer_dict, clauses):
    #         ratio_sat += 1
    #         assert (evaluate_cnf_formula(answer_dict, or_result_vars, bqm) == 0.0)
    #     else:
    #         assert (evaluate_cnf_formula(answer_dict, or_result_vars, bqm) > 0.0)
    # print("ratio sat:",round(float(ratio_sat) / float(2 ** len(variables_1)), 2))
    return bqm, clauses



