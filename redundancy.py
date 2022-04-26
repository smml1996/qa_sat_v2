from typing import Dict, Set, List, Any, Tuple
from dimod import BinaryQuadraticModel, Vartype
from dwave.system import DWaveSampler

from utils import evaluate_cnf_formula, matriarch8, get_qubits_values
from embedding_utils import get_qpu
from neal import SimulatedAnnealingSampler


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
def get_valid_edges(qpu, vars1, vars2) -> List[Tuple[int, int]]:
    valid_edges = []
    for v1 in vars1:
        for v2 in vars2:
            if does_edge_exists(qpu, v1, v2):
                valid_edges.append((v1, v2))
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
        weight = bqm.linear[logic_variable] / len(chain)
        for c in chain:
            new_qubo.add_variable(c, weight)

    for ((a, b), coupler) in bqm.quadratic.items():
        valid_edges = get_valid_edges(qpu, embedding[a], embedding[b])
        weight = coupler / len(valid_edges)
        for (v1, v2) in valid_edges:
            new_qubo.add_interaction(v1, v2, weight)
    new_qubo.offset = bqm.offset
    return new_qubo


def get_edge_weight(bqm, edge):
    v1 = edge[0]
    v2 = edge[1]
    try:
        assert(bqm.adj[v1][v2] == bqm.adj[v2][v1])
        return abs(bqm.adj[v1][v2])
    except:
        return None


def find_edge_in_embedding(bqm, l1: int, l2: int, embedding: Dict[int, List[int]], qpu: DWaveSampler):
    p_l1 = embedding[l1]
    p_l2 = embedding[l2]

    valid_edges = get_valid_edges(qpu, p_l1, p_l2)
    assert(len(valid_edges) > 0)

    min_value = get_edge_weight(bqm, valid_edges[0])
    current_min_index = 0
    for (index,edge) in enumerate(valid_edges):
        weight = get_edge_weight(bqm, edge)
        if weight is not None:
            if weight < min_value:
                min_value = weight
                current_min_index = index

    answer_edge = valid_edges[current_min_index]
    p1 = answer_edge[0]
    p2 = answer_edge[1]
    assert (p1 != p2 )
    return p1, p2

def get_linear_var_in_embedding(bqm, embedding, l1):
    p_vars = embedding[l1]

    min_val = 0
    min_val_index = -1
    for (index, p_var) in enumerate(p_vars):
        if p_var in bqm.variables:
            if min_val_index == -1 :
                min_val = abs(bqm.linear[p_var])
                min_val_index = index
            elif min_val >  abs(bqm.linear[p_var]):
                min_val = abs(bqm.linear[p_var])
                min_val_index = index

    if min_val_index == -1:
        min_val_index = 0

    return p_vars[min_val_index]





def emb_logic_or(bqm, or_dict, embedding, qpu, temp_x1, temp_x2, is_res_fixed=False, clause_index=None):

    if clause_index == None:
        raise Exception("clause index is None")

    x1 = get_linear_var_in_embedding(bqm, embedding, abs(temp_x1))
    x2 = get_linear_var_in_embedding(bqm, embedding, abs(temp_x2))

    if is_res_fixed:
        res = "a1"
        x12, x21 = find_edge_in_embedding(bqm, abs(temp_x1), abs(temp_x2), embedding, qpu)
        x13 = x1
        x31 = res
        x23 = x2
        x32 = res
        temp_res = "a1"
    else:
        if temp_x1 < temp_x2:
            temp_res = or_dict[clause_index][(temp_x1, temp_x2)]
        else:
            temp_res = or_dict[clause_index][(temp_x2, temp_x1)]
        res = get_linear_var_in_embedding(bqm, embedding, temp_res)
        x12, x21 = find_edge_in_embedding(bqm, abs(temp_x1), abs(temp_x2), embedding, qpu)
        x13, x31 = find_edge_in_embedding(bqm, abs(temp_x1), temp_res, embedding, qpu)
        x23, x32 = find_edge_in_embedding(bqm, abs(temp_x2), temp_res, embedding, qpu)

    if temp_x1 > 0:
        if temp_x2 > 0:
            assert (temp_x1 > 0 and temp_x2 > 0)
            # perform traditional logic or
            bqm.add_variable(x1, 2)
            bqm.add_variable(x2, 2)
            bqm.add_variable(res, 2)

            bqm.add_interaction(x12, x21, 2)
            bqm.add_interaction(x13, x31, -4)
            bqm.add_interaction(x23, x32, -4)
            # logic or has offset 0
        else:
            assert (temp_x1 > 0 and temp_x2 < 0)
            # compute truth table with negated value of x2 (MATRIARCH8)
            prev_offset = bqm.offset
            matriarch8(bqm, x1, x2, res, x12, x21, x13, x31, x23, x32)
            assert (prev_offset + 2 == bqm.offset)
    elif temp_x2 > 0:
        assert (temp_x1 < 0 and temp_x2 > 0)
        # compute truth table with negated value of x1 (MATRIARCH8)
        prev_offset = bqm.offset
        matriarch8(bqm, x2, x1, res, x12, x21, x13, x31, x23, x32)
        assert (prev_offset + 2 == bqm.offset)
    else:
        assert (temp_x1 < 0 and temp_x2 < 0)
        # compute NAND
        bqm.add_variable(x1, -4 / 2)
        bqm.add_variable(x2, -4 / 2)
        bqm.add_variable(res, -6 / 2)
        bqm.add_interaction(x12, x21, 2 / 2)
        bqm.add_interaction(x13, x31, 4 / 2)
        bqm.add_interaction(x23, x32, 4 / 2)
        bqm.offset += 6 / 2

    return res, temp_res


def emb_clause_to_bqm(bqm: BinaryQuadraticModel, or_dict, embedding, qpu, clause: list[int],
                      clause_index: int):

    if len(clause) == 0:
        raise Exception("clause has length 0")

    temp, temp_res = emb_logic_or(bqm, or_dict, embedding, qpu, clause[0], clause[1], len(clause) == 2, clause_index)
    for (i,x) in enumerate(clause[2:]):
        is_res_fixed = (i == len(clause)-3)
        prev = temp_res
        temp, temp_res = emb_logic_or(bqm, or_dict, embedding, qpu, prev, x, is_res_fixed, clause_index)

    bqm.fix_variable(temp, 1)


def get_or_dict(or_result_vars: Dict[int, Any], res_vars_to_clause_index: Dict[int, int], num_clauses: int):
    answer = dict()

    for i in range(num_clauses):
        answer[i] = dict()

    for (res_var, (_, i1, i2)) in or_result_vars.items():
        assert(i1 != i2)
        assert(res_var in res_vars_to_clause_index.keys())

        clause_index = res_vars_to_clause_index[res_var]
        if i1 < i2:
            key = (i1, i2)
        else:
            key = (i2, i1)
        assert(key not in answer[clause_index].keys())
        answer[clause_index][key] = res_var
    return answer

def emb_cnf_to_bqm(embedding: Dict[int, List[int]], or_result_vars, num_variables, variables, clauses,
                   res_vars_to_clause_index = None):
    if res_vars_to_clause_index == None:
        raise Exception("no clauses indices")

    assert (num_variables == len(variables))
    qpu = get_qpu()
    bqm: BinaryQuadraticModel = BinaryQuadraticModel(Vartype.BINARY)

    or_dict = get_or_dict(or_result_vars, res_vars_to_clause_index, len(clauses))

    for (index, clause) in enumerate(clauses):
        emb_clause_to_bqm(bqm, or_dict, embedding, qpu, clause, index)
    return bqm


def emb_evaluate_cnf_formula(embedding: dict[int, List[int]], or_vars, answer, bqm, bqm2 ):
    all_values = get_qubits_values(answer, or_vars, bqm)

    emb_values = dict()

    for (logic_var, chain) in embedding.items():
        for c in chain:
            emb_values[c] = all_values[logic_var]
    return bqm2.energy(emb_values)