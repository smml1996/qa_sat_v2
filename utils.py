from typing import Dict, List, Tuple, Set, Any
from enum import Enum
from dimod import BinaryQuadraticModel, Vartype
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from greedy import SteepestDescentComposite
from dwave.preprocessing.lower_bounds import roof_duality

class GateType(Enum):
    OR = 1
    NAND = 2
    MATRIARCH8 = 3
    MATRIARCH8_FLIPPED = 4

def evaluate_clause(result: Dict[int, int], clause: List[int]) -> bool:
    clause_value = False
    for x in clause:
        val = bool(result[abs(x)])
        if x < 0:
            val = not val
        clause_value = clause_value or val
    return clause_value

def evaluate_clauses(result, clauses):
    answer = True
    for clause in clauses:
        answer = answer and evaluate_clause(result, clause)
    return answer

def get_false_clauses(result, clauses):
    answer = []
    for clause in clauses:
        if not evaluate_clause(result, clause):
            answer.append(clause)
    return answer

def create_all_variables(variables: Set[int]) -> BinaryQuadraticModel:
    bqm = BinaryQuadraticModel(Vartype.BINARY)
    for var in variables:
        bqm.add_variable(var)
    return bqm

def matriarch8(bqm: BinaryQuadraticModel, x1: int, x2: int, res: int,x12, x21, x13, x31, x23, x32) -> None:
    assert (x1 > 0 and x2 > 0)
    bqm.add_variable(x1, 4)
    bqm.add_variable(x2, -2)
    bqm.add_variable(res, -2)

    bqm.add_interaction(x12, x21, -2)
    bqm.add_interaction(x13, x31, -4)
    bqm.add_interaction(x23, x32, 4)

    bqm.offset += 2

def get_bqm_size(bqm: BinaryQuadraticModel) -> int:
    return len(bqm.variables)

def get_ancilla_index(bqm: BinaryQuadraticModel) -> int:
    return get_bqm_size(bqm) + 1

def logic_or(bqm: BinaryQuadraticModel, variables: Set[int], x1: int, x2: int) -> (int, GateType):
    assert (x1 != 0 and x2 != 0)

    res: int = get_ancilla_index(bqm)
    variables.add(res)

    if x1 > 0:
        if x2 > 0:
            assert(x1 > 0 and x2 > 0)
            # perform traditional logic or
            bqm.add_variable(x1, 2)
            bqm.add_variable(x2, 2)
            bqm.add_variable(res, 2)

            bqm.add_interaction(x1, x2, 2)
            bqm.add_interaction(x1, res, -4)
            bqm.add_interaction(x2, res, -4)
            # logic or has offset 0
            gate_type = GateType.OR
        else:
            assert(x1 > 0 and x2 < 0)
            # compute truth table with negated value of x2 (MATRIARCH8)
            prev_offset = bqm.offset
            matriarch8(bqm, x1, abs(x2), res)
            assert (prev_offset + 2 == bqm.offset)
            gate_type = GateType.MATRIARCH8
    elif x2 > 0:
        assert(x1 < 0 and x2 > 0)
        # compute truth table with negated value of x1 (MATRIARCH8)
        prev_offset = bqm.offset
        matriarch8(bqm, x2, abs(x1), res)
        assert(prev_offset + 2 == bqm.offset)
        gate_type = GateType.MATRIARCH8_FLIPPED
    else:
        assert(x1 < 0 and x2 < 0)
        # compute NAND
        bqm.add_variable(abs(x1), -4)
        bqm.add_variable(abs(x2), -4)
        bqm.add_variable(res, -6)
        bqm.add_interaction(abs(x1), abs(x2), 2)
        bqm.add_interaction(abs(x1), res, 4)
        bqm.add_interaction(abs(x2), res, 4)
        bqm.offset += 6
        gate_type = GateType.NAND

    return res, gate_type

def only_one_true(vars) -> BinaryQuadraticModel:
    assert(len(vars) == 5)
    bqm = BinaryQuadraticModel(Vartype.BINARY)

    bqm.offset = 2

    for var in vars:
        bqm.add_variable(abs(var), -2)

    for i in range(len(vars)):
        for j in range(i+1,len(vars)):
            bqm.add_interaction(abs(vars[i]), abs(vars[j]),4)
    return bqm

def clause_to_bqm(bqm: BinaryQuadraticModel, variables: Set[int], clause: List[int], clause_index=None) \
        -> Tuple[Any, Dict[int, Tuple[GateType, int, int]], Dict[int, int]]:

    if clause_index == None:
        raise Exception("clause index is None")

    if len(clause) == 0:
        raise Exception("clause has length 0")


    or_result_vars = dict()
    res_vars_to_clause_index = dict()

    if len(clause) == 5:
        bqm.update(only_one_true(clause))
        return None, or_result_vars, res_vars_to_clause_index

    temp, gate_type = logic_or(bqm, variables, clause[0], clause[1])
    or_result_vars[temp] = (gate_type, clause[0], clause[1])
    res_vars_to_clause_index[temp] = clause_index

    for x in clause[2:]:
        prev = temp
        temp, gate_type = logic_or(bqm, variables, prev, x)
        or_result_vars[temp] = (gate_type, prev, x)
        res_vars_to_clause_index[temp] = clause_index

    # the clause should evaluate to true
    bqm.fix_variable(temp, 1)
    ## fix_variable(bqm, variables, temp, True)

    return temp, or_result_vars, res_vars_to_clause_index

def cnf_to_bqm(variables: Set[int], clauses: List[List[int]]) \
        -> Tuple[BinaryQuadraticModel, Dict[int, Tuple[GateType, int, int]], List[int], Dict[int, int]]:
    bqm: BinaryQuadraticModel = create_all_variables(variables)
    or_result_vars: Dict[int, Tuple[GateType, int, int]] = dict()
    res_vars_to_clause_index: Dict[int, int] = dict()

    clauses_qubits: List[int] = []
    for (index, clause) in enumerate(clauses):
        clause_qubit, temp_or_result_vars, temp_res_vars_to_clause = clause_to_bqm(bqm, variables, clause, index)
        if clause_qubit is not None:
            clauses_qubits.append(clause_qubit)
        or_result_vars.update(temp_or_result_vars)
        res_vars_to_clause_index.update(temp_res_vars_to_clause)

    return bqm, or_result_vars, clauses_qubits, res_vars_to_clause_index


def load_cnf(path: str) -> Tuple[int, int, Set[int], List[List[int]]]:
    clauses: List[List[int]] = []
    variables: Set[int] = set()
    file = open(path)
    num_variables = -1
    num_clauses = -1
    for line in file.readlines():
        line = line[:-1]
        elements = line.split(" ")
        if elements[0] == "c":
            # this is a comment
            continue
        elif elements[0] == "p":
            assert (len(elements) == 4)
            assert (elements[1] == "cnf")
            num_variables = int(elements[2])
            num_clauses = int(elements[3])
        else:
            assert (int(elements[-1]) == 0)
            clause = []
            for x in elements[:-1]:
                int_x = int(x)
                assert(abs(int_x) <= num_variables)
                clause.append(int_x)
                variables.add(abs(int_x))
            clauses.append(clause)
    assert (num_variables != -1 and num_clauses != -1)
    return num_variables, num_clauses, variables, clauses


def evaluate_gate(gate_type: GateType, x1: bool, x2: bool) -> bool:
    if gate_type == GateType.OR:
        return x1 or x2
    if gate_type == GateType.NAND:
        return not (x1 and x2)
    else:
        x = x1
        negated = x2
        if gate_type == GateType.MATRIARCH8_FLIPPED:
            x = x2
            negated = x1
        return x or (not negated)


def get_qubits_values(values, or_gates, bqm):
    all_values = values.copy()
    for i in bqm.variables:
        if i not in all_values.keys():
            gate_type, x1, x2 = or_gates[i]
            all_values[i] = int(evaluate_gate(gate_type, bool(all_values[abs(x1)]), bool(all_values[abs(x2)])))

    return all_values
def evaluate_cnf_formula(values: Dict[int, int], or_gates: Dict[int, Tuple[GateType, int, int]], bqm: BinaryQuadraticModel) -> float:

    all_values = get_qubits_values(values, or_gates, bqm)
    assert len(bqm.variables) == len(all_values.keys())
    return bqm.energy(all_values)


def get_greedy_quantum_sampler(embedding=None):
    sampler = SteepestDescentComposite(
        FixedEmbeddingComposite(DWaveSampler(solver={"name": "Advantage_system4.1"}), embedding))
    return sampler

def minimize_qubo(bqm):
    return roof_duality(bqm)


def get_avg_energy(sampleset):
    energies = []
    for s in sampleset.record:
        energies.append(s[1])
    return round(sum(energies)/len(energies), 2)

def only_one_true(vars) -> BinaryQuadraticModel:
    assert(len(vars) == 5)
    bqm = BinaryQuadraticModel(Vartype.BINARY)

    bqm.offset = 2

    for var in vars:
        bqm.add_variable(var, -2)

    for i in range(len(vars)):
        for j in range(i+1,len(vars)):
            bqm.add_interaction(vars[i], vars[j],4)
    return bqm