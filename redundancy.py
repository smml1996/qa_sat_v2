from typing import Dict, Set, List
from dimod import BinaryQuadraticModel, Vartype
from dwave.system import DWaveSampler

from utils import evaluate_cnf_formula, matriarch8
from embedding_utils import get_qpu


def initialize_mirror_vars(variables: Set[int], mirrors: Dict[int, int],
                           original_vars_to_mirrors: Dict[int, List[int]]):
    for var in variables:
        mirrors[var] = var
        original_vars_to_mirrors[var] = []


def get_key_with_value(d, val):
    for (key, value) in d.items():
        if d[key] == val:
            return key

    return -1


def get_count_occurrences(key, clauses):
    ans = 0
    for clause in clauses:
        for var in clause:
            if abs(var) == key:
                ans += 1
    return ans


def get_original_variable(var: int, mirrors: Dict[int, int]):
    assert (var in mirrors)
    if mirrors[var] == var:
        return var
    return get_original_variable(mirrors[var], mirrors)


def mirror_variable(key, variables, clauses, mirrors: Dict[int, int],
                    original_vars_to_mirrors: Dict[int, List[int]]) -> (int, int):
    assert key > 0

    replacements = get_count_occurrences(key, clauses) // 2
    if replacements == 0:
        return -1
    new_variable = len(variables) + 1
    assert (new_variable not in variables)
    variables.add(new_variable)

    mirrors[new_variable] = key
    original_var = get_original_variable(key, mirrors)
    assert (original_var in original_vars_to_mirrors.keys())
    original_vars_to_mirrors[original_var].append(new_variable)

    for clause in clauses:
        for i in range(len(clause)):
            if abs(clause[i]) == key:
                if clause[i] < 0:
                    clause[i] = -new_variable
                else:
                    clause[i] = new_variable
                replacements -= 1
                if replacements == 0:
                    return new_variable
    raise Exception("this part of the code should not occur")


def get_variable_with_less_entanglements(bqm, original_variable, copies):
    min_value = len(bqm.adjacency[original_variable])
    variable_to_copy = original_variable

    for c in copies:
        if c in bqm.adjacency.keys():
            l = len(bqm.adjacency[c])
            if l < min_value:
                min_value = l
                variable_to_copy = c
    return variable_to_copy


def update_bqm_mirror_variables(bqm: BinaryQuadraticModel, mirrors: Dict[int, int],
                                original_vars_to_mirrors: Dict[int, List[int]]):
    connected_copies = dict()
    for key in original_vars_to_mirrors.keys():
        connected_copies[key] = [key]

    for (mirror, value) in mirrors.items():
        if mirrors[mirror] != mirror:
            original_variable = get_original_variable(mirror, mirrors)
            variable_to_copy = get_variable_with_less_entanglements(bqm, original_variable,
                                                                    connected_copies[original_variable])
            connected_copies[original_variable].append(mirror)
            bqm.add_variable(variable_to_copy, 2)
            bqm.add_variable(mirror, 2)
            bqm.add_interaction(variable_to_copy, mirror, -4)
    return connected_copies


def update_bqm_single_mirror(bqm, var, mirrors, original_vars_to_mirrors):
    connected_copies = dict()
    for key in original_vars_to_mirrors.keys():
        connected_copies[key] = [key]
    for mirror in original_vars_to_mirrors[var]:
        if mirrors[mirror] != mirror:
            original_variable = get_original_variable(mirror, mirrors)
            variable_to_copy = get_variable_with_less_entanglements(bqm, original_variable,
                                                                    connected_copies[original_variable])
            connected_copies[original_variable].append(mirror)
            bqm.add_variable(variable_to_copy, 2)
            bqm.add_variable(mirror, 2)
            bqm.add_interaction(variable_to_copy, mirror, -4)
    return connected_copies


def evaluate_bqm(bqm, sample, variables, or_result_vars, original_vars_to_mirrors):
    values = dict()
    for var in variables:
        value = sample[var]
        values[var] = value
        for var2 in original_vars_to_mirrors[var]:
            values[var2] = value
    return evaluate_cnf_formula(values, or_result_vars, bqm)


def get_bqm_chain_lengths(bqm):
    answer = []
    for (key, edges) in bqm.adjacency.items():
        answer.append(len(edges))

    return answer


def does_edge_exists(qpu, a, b):
    assert (a in qpu.adjacency.keys())
    for q in qpu.adjacency[a]:
        if q == b:
            return True
    return False


def get_ancilla_to_connect(qpu, physical_var, bqm, chain):
    answer = -1

    for c in chain:
        if does_edge_exists(qpu, physical_var, c):
            if answer == -1:
                answer = c
            else:
                if len(bqm.adj[c]) < len(bqm.adj[answer]):
                    answer = c
    assert (answer != -1)
    return answer


####### transforming embedding #######
def get_valid_edges(qpu, vars1, vars2):
    valid_edges = set()
    for v1 in vars1:
        for v2 in vars2:
            if does_edge_exists(qpu, v1, v2):
                if v1 < v2:
                    e1 = v1
                    e2 = v2
                else:
                    e1 = v2
                    e2 = v1
                valid_edges.add((e1, e2))
    return valid_edges


def remove_chains(embedding):
    qpu = get_qpu()
    new_embedding = {}
    new_qubo = BinaryQuadraticModel(Vartype.BINARY)
    for (logic_variable, chain) in embedding.items():
        connected_components = []
        chain_length = len(chain)
        new_embedding[chain[0]] = [chain[0]]
        connected_components.append([chain[0]])
        alpha = 2
        alpha_c = -2 * alpha

        for i in range(0, chain_length):
            new_embedding[chain[i]] = [chain[i]]
            new_qubo.add_variable(chain[i])

        edges = get_valid_edges(qpu, chain, chain)

        for (a, b) in edges:
            assert (a != b)
            new_qubo.add_variable(a, alpha)
            new_qubo.add_variable(b, alpha)
            new_qubo.add_interaction(a, b, alpha_c)

    return new_qubo, new_embedding


def get_embedded_bqm(bqm, embedding):
    qpu = get_qpu()
    new_qubo = BinaryQuadraticModel(Vartype.BINARY)

    for (logic_variable, chain) in embedding.items():
        weight = round(bqm.linear[logic_variable] / len(chain), 0)
        for c in chain:
            new_qubo.add_variable(c, weight)

    for ((a, b), coupler) in bqm.quadratic.items():
        valid_edges = get_valid_edges(qpu, embedding[a], embedding[b])
        weight = round(coupler / len(valid_edges), 0)
        for (v1, v2) in valid_edges:
            new_qubo.add_interaction(v1, v2, weight)
    new_qubo.offset = bqm.offset
    return new_qubo


def get_valid_edges_variables(qpu: DWaveSampler, p_l1, p_l2):
    valid_vars1 = set()
    valid_vars2 = set()

    for v1 in p_l1:
        for v2 in p_l2:
            if does_edge_exists(qpu, v1, v2):
                valid_vars1.add(v1)
                valid_vars2.add(v2)
    return valid_vars1, valid_vars2


def find_variables_in_embedding3(l1: int, l2: int, l3: int, embedding: Dict[int, List[int]], qpu: DWaveSampler):
    print("vars:", l1, l2, l3)
    p_l1 = embedding[l1]
    p_l2 = embedding[l2]
    p_l3 = embedding[l3]
    print(p_l1)
    print(p_l2)
    print(p_l3)

    valid_qubits1, valid_qubits2 = get_valid_edges_variables(qpu, p_l1, p_l2)
    temp_valid_qubits1, valid_qubits3 = get_valid_edges_variables(qpu, p_l1, p_l3)
    temp_valid_qubits2, temp_valid_qubits3 = get_valid_edges_variables(qpu, p_l2, p_l3)

    print("valid qubits: ", valid_qubits1)
    print("t_valid qubits: ", temp_valid_qubits1)
    valid_qubits1 = valid_qubits1.intersection(temp_valid_qubits1)

    print("intersection valid_qubits", valid_qubits1)
    assert (len(valid_qubits1) > 0)

    valid_qubits2 = valid_qubits2.intersection(temp_valid_qubits2)
    assert (len(valid_qubits2) > 0)

    valid_qubits3 = valid_qubits3.intersection(temp_valid_qubits3)
    assert (len(valid_qubits3) > 0)

    p1 = valid_qubits1.pop()
    p2 = valid_qubits2.pop()
    p3 = valid_qubits3.pop()

    assert (p1 != p2 and p1 != p3 and p2 != p3)
    return p1, p2, p3


def find_variables_in_embedding2(l1: int, l2: int, embedding: Dict[int, List[int]], qpu: DWaveSampler):
    p_l1 = embedding[l1]
    p_l2 = embedding[l2]

    valid_qubits1, valid_qubits2 = get_valid_edges_variables(qpu, p_l1, p_l2)

    p1 = valid_qubits1.pop()
    p2 = valid_qubits2.pop()

    assert (p1 != p2 )
    return p1, p2

def emb_logic_or(bqm, or_dict, variables, embedding, qpu, temp_x1, temp_x2, is_res_fixed=False):
    if temp_x1 < temp_x2:
        temp_res = or_dict[(temp_x1, temp_x2)]
    else:
        temp_res = or_dict[(temp_x2, temp_x1)]


    if is_res_fixed:
        x1,x2 = find_variables_in_embedding2(abs(temp_x1), abs(temp_x2), embedding, qpu)
        res = str(temp_res) + "a"
    else:
        x1, x2, res = find_variables_in_embedding3(abs(temp_x1), abs(temp_x2), temp_res, embedding, qpu)
    if x1 > 0:
        if x2 > 0:
            assert (x1 > 0 and x2 > 0)
            # perform traditional logic or
            bqm.add_variable(x1, 2)
            bqm.add_variable(x2, 2)
            bqm.add_variable(res, 2)

            bqm.add_interaction(x1, x2, 2)
            bqm.add_interaction(x1, res, -4)
            bqm.add_interaction(x2, res, -4)
            # logic or has offset 0
        else:
            assert (x1 > 0 and x2 < 0)
            # compute truth table with negated value of x2 (MATRIARCH8)
            prev_offset = bqm.offset
            matriarch8(bqm, x1, abs(x2), res)
            assert (prev_offset + 2 == bqm.offset)
    elif x2 > 0:
        assert (x1 < 0 and x2 > 0)
        # compute truth table with negated value of x1 (MATRIARCH8)
        prev_offset = bqm.offset
        matriarch8(bqm, x2, abs(x1), res)
        assert (prev_offset + 2 == bqm.offset)
    else:
        assert (x1 < 0 and x2 < 0)
        # compute NAND
        bqm.add_variable(abs(x1), -4 / 2)
        bqm.add_variable(abs(x2), -4 / 2)
        bqm.add_variable(res, -6 / 2)
        bqm.add_interaction(abs(x1), abs(x2), 2 / 2)
        bqm.add_interaction(abs(x1), res, 4 / 2)
        bqm.add_interaction(abs(x2), res, 4 / 2)
        bqm.offset += 6 / 2

    return res


def emb_clause_to_bqm(bqm, or_dict, embedding, qpu, variables, clause):
    print(clause)
    if len(clause) == 0:
        raise Exception("clause has length 0")

    temp = emb_logic_or(bqm, or_dict, variables, embedding, qpu, clause[0], clause[1], len(clause) == 2)

    for (i,x) in enumerate(clause[2:]):
        is_res_fixed = (i == len(clause)-1)
        prev = temp
        temp = emb_logic_or(bqm, or_dict, variables, embedding, qpu, prev, x, is_res_fixed)


    bqm.fix_variable(temp, 1)


def get_or_dict(or_result_vars):
    answer = dict()
    for (res_var, (_, i1, i2)) in or_result_vars.items():
        print("or_dict:",res_var, i1, i2)
        assert(i1 != i2)
        assert ((i1, i2) not in answer.keys())
        assert ((i2, i1) not in answer.keys())
        if i1 < i2:
            answer[(i1,i2)] = res_var
        else:
            answer[(i2, i1)] = res_var
    return answer

def emb_cnf_to_bqm(embedding, or_result_vars, num_variables, variables, clauses):
    assert (num_variables == len(variables))
    qpu = get_qpu()
    bqm: BinaryQuadraticModel = BinaryQuadraticModel(Vartype.BINARY)
    or_dict = get_or_dict(or_result_vars)

    for clause in clauses:
        emb_clause_to_bqm(bqm, or_dict, embedding, qpu, variables, clause)
    return bqm
