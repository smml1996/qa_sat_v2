from typing import Dict, Set, List
from dimod import BinaryQuadraticModel, Vartype
from utils import evaluate_cnf_formula, get_ancilla_index
from embedding_utils import get_qpu


def initialize_mirror_vars(variables: Set[int], mirrors: Dict[int, int], original_vars_to_mirrors: Dict[int, List[int]]):
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


def mirror_variable(key, variables, clauses, mirrors: Dict[int, int], original_vars_to_mirrors: Dict[int, List[int]]) -> (int, int):
    assert key > 0

    replacements = get_count_occurrences(key, clauses) // 2
    if replacements == 0:
        return -1
    new_variable = len(variables) + 1
    assert (new_variable not in variables)
    variables.add(new_variable)

    mirrors[new_variable] = key
    original_var = get_original_variable(key, mirrors)
    assert(original_var in original_vars_to_mirrors.keys())
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

def update_bqm_mirror_variables(bqm: BinaryQuadraticModel, mirrors: Dict[int, int], original_vars_to_mirrors: Dict[int, List[int]]):
    connected_copies = dict()
    for key in original_vars_to_mirrors.keys():
        connected_copies[key] = [key]

    for (mirror, value) in mirrors.items():
        if mirrors[mirror] != mirror:
            original_variable = get_original_variable(mirror, mirrors)
            variable_to_copy = get_variable_with_less_entanglements(bqm, original_variable, connected_copies[original_variable])
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

def does_edge_exists(qpu, a , b):
    assert(a in qpu.adjacency.keys())
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
    assert(answer != -1)
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
        alpha_c = -2*alpha

        for i in range(0, chain_length):
            new_embedding[chain[i]] = [chain[i]]
            new_qubo.add_variable(chain[i])

        edges = get_valid_edges(qpu, chain, chain)

        for (a,b) in edges:
            assert(a!=b)
            new_qubo.add_variable(a, alpha)
            new_qubo.add_variable(b, alpha)
            new_qubo.add_interaction(a, b, alpha_c)

    return new_qubo, new_embedding

def get_embedded_bqm(bqm, embedding):
    qpu = get_qpu()
    new_qubo = BinaryQuadraticModel(Vartype.BINARY)

    for (logic_variable, chain) in embedding.items():
        weight = bqm.linear[logic_variable]
        for c in chain:
            new_qubo.add_variable(c, weight)

    for ((a,b),coupler) in bqm.quadratic.items():
        valid_edges = get_valid_edges(qpu, embedding[a], embedding[b])
        weight = coupler
        for (v1,v2) in valid_edges:
            new_qubo.add_interaction(v1, v2, weight)
    new_qubo.offset = bqm.offset
    return new_qubo


