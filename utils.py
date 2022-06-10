from typing import Dict, List, Tuple, Set, Any
from enum import Enum
from dimod import BinaryQuadraticModel, Vartype
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from greedy import SteepestDescentComposite
from dwave.preprocessing.lower_bounds import roof_duality
from statistics import variance


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


def matriarch8(bqm: BinaryQuadraticModel, x1: int, x2: int, res: int) -> None:
    assert (x1 > 0 and x2 > 0)
    bqm.add_variable(x1, 4)
    bqm.add_variable(x2, -2)
    bqm.add_variable(res, -2)

    bqm.add_interaction(x1, x2, -2)
    bqm.add_interaction(x1, res, -4)
    bqm.add_interaction(x2, res, 4)

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
            assert (x1 > 0 and x2 > 0)
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
            assert (x1 > 0 > x2)
            # compute truth table with negated value of x2 (MATRIARCH8)
            prev_offset = bqm.offset
            matriarch8(bqm, x1, abs(x2), res)
            assert (prev_offset + 2 == bqm.offset)
            gate_type = GateType.MATRIARCH8
    elif x2 > 0:
        assert (x1 < 0 < x2)
        # compute truth table with negated value of x1 (MATRIARCH8)
        prev_offset = bqm.offset
        matriarch8(bqm, x2, abs(x1), res)
        assert (prev_offset + 2 == bqm.offset)
        gate_type = GateType.MATRIARCH8_FLIPPED
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
        gate_type = GateType.NAND

    return res, gate_type


def only_one_true(vars_) -> BinaryQuadraticModel:
    bqm = BinaryQuadraticModel(Vartype.BINARY)

    bqm.offset = 2 / 2

    for var in vars_:
        bqm.add_variable(abs(var), -2 / 2)

    for i in range(len(vars_)):
        for j in range(i + 1, len(vars_)):
            bqm.add_interaction(abs(vars_[i]), abs(vars_[j]), 4 / 2)
    return bqm


def clause_to_bqm(bqm: BinaryQuadraticModel, variables: Set[int], clause: List[int], clause_index=None) \
        -> Tuple[Any, Dict[int, Tuple[GateType, int, int]], Dict[int, int]]:
    if clause_index == None:
        raise Exception("clause index is None")

    if len(clause) == 0:
        raise Exception("clause has length 0")

    or_result_vars = dict()
    res_vars_to_clause_index = dict()

    # if len(clause) > 2:
    #     bqm.update(only_one_true(clause))
    #     return None, or_result_vars, res_vars_to_clause_index

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
                assert (abs(int_x) <= num_variables)
                clause.append(int_x)
                variables.add(abs(int_x))
            clauses.append(clause)
    file.close()
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


def evaluate_cnf_formula(values: Dict[int, int], or_gates: Dict[int, Tuple[GateType, int, int]],
                         bqm: BinaryQuadraticModel) -> float:
    all_values = get_qubits_values(values, or_gates, bqm)
    # assert len(bqm.variables) == len(all_values.keys())
    return bqm.energy(all_values)


def get_greedy_quantum_sampler(embedding=None):
    sampler = SteepestDescentComposite(
        FixedEmbeddingComposite(DWaveSampler(solver={"name": "Advantage_system4.1"}), embedding))
    return sampler


def get_quantum_sampler(embedding=None, name="Advantage_system4.1"):
    sampler = FixedEmbeddingComposite(DWaveSampler(solver={"name": name}), embedding)
    return sampler


def sample_with_sampler(embedding, bqm, num_reads_, chain_strengh_, clauses, name="Advantage_system4.1"):
    sampler = get_quantum_sampler(embedding, name)
    sampleset = sampler.sample(bqm, num_reads=num_reads_, chain_strength=chain_strengh_, auto_scale=True)

    lowest_energy = sampleset.first.energy
    print("lowest energy achieved:", lowest_energy)

    if lowest_energy == 0.0:
        print(f"ground energy samples {len(sampleset.lowest())}/{num_reads_}")
        for sample in sampleset.lowest():
            assert (bqm.energy(sample) == 0.0)
            assert (evaluate_clauses(sample, clauses))
        return round(len(sampleset.lowest()) / num_reads_, 2)
    else:
        print(f"ground energy samples 0/{num_reads_}")
        return 0


def minimize_qubo(bqm, _strict=True):
    return roof_duality(bqm, strict=_strict)


def get_avg_energy(sampleset):
    energies = []
    for s in sampleset.record:
        energies.append(s[1])
    return round(sum(energies) / len(energies), 2)


def process_unicorn_file(path: str) -> BinaryQuadraticModel:
    bqm = BinaryQuadraticModel(Vartype.BINARY)
    file = open(path)
    section = 0
    for line in file.readlines():
        line = line.replace("\n", "")

        if line == "":
            section += 1
        else:

            elements = line.split(" ")
            if section == 0:
                assert (len(elements) == 2)
                bqm.offset = float(elements[1])
            elif section == 1:
                assert (len(elements) == 3)
            elif section == 2:
                assert (len(elements) == 3)
            elif section == 3:
                assert (len(elements) == 2)
                var = int(elements[0])
                bias = float(elements[1])
                bqm.add_variable(var, bias)
            elif section == 4:
                assert (len(elements) == 3)
                var1 = int(elements[0])
                var2 = int(elements[1])
                coupler = float(elements[2])
                bqm.add_interaction(var1, var2, coupler)
    return bqm


def fix_constant_integer(bqm, variables, n):
    for var in variables:
        bqm.fix_variable(var, n % 2)
        n /= 2


def get_long_clauses(threshold, clauses):
    answer = []
    for clause in clauses:
        if len(clause) >= threshold:
            answer.append(clause)
    return answer


def xor_variables(num_variables, var1, var2):
    current_var = num_variables
    current_var += 1
    z = current_var
    answer = [[var1, var2, -z], [var1, -var2, z], [-var1, var2, z], [-var1, -var2, -z]]
    return current_var, answer


# def xor_clause(num_variables, clause):
#     assert(len(clause) > 2)
#     current_var = num_variables
#     prev_result = clause[0]
#     answer = []
#     new_variables_mapping = dict()
#     for i in range(1, len(clause)):
#         current_var, new_clauses = xor_variables(current_var, prev_result, clause[i])
#         new_variables_mapping[current_var] = (prev_result, clause[i])
#         answer.extend(new_clauses)
#     return current_var, answer, new_variables_mapping

def xor_clause(num_variables, clause):
    x4 = clause[0]
    x3 = clause[1]
    x2 = clause[2]
    x1 = clause[3]
    x0 = clause[4]
    answer = [[x4, x3, x2, -x1, -x0], [x4, x3, -x2, x1, -x0], [x4, x3, -x2, -x1, x0], [x4, x3, -x2, -x1, -x0],
              [x4, -x3, x2, x1, -x0], [x4, -x3, x2, -x1, x0], [x4, -x3, x2, -x1, -x0], [x4, -x3, -x2, x1, x0],
              [x4, -x3, -x2, x1, -x0], [x4, -x3, -x2, -x1, x0], [x4, -x3, -x2, -x1, -x0], [-x4, x3, x2, x1, -x0],
              [-x4, x3, x2, -x1, x0], [-x4, x3, x2, -x1, -x0], [-x4, x3, -x2, x1, x0], [-x4, x3, -x2, x1, -x0],
              [-x4, x3, -x2, -x1, x0], [-x4, x3, -x2, -x1, -x0], [-x4, -x3, x2, x1, x0], [-x4, -x3, x2, x1, -x0],
              [-x4, -x3, x2, -x1, x0], [-x4, -x3, x2, -x1, -x0], [-x4, -x3, -x2, x1, x0], [-x4, -x3, -x2, x1, -x0],
              [-x4, -x3, -x2, -x1, x0], [-x4, -x3, -x2, -x1, -x0]]

    # (x4 ∨ x3 ∨ x2 ∨ x1 ∨ x0)
    # answer.append(clause)

    # (x4 ∨ x3 ∨ x2 ∨ ¬x1 ∨ ¬x0)

    # (x4 ∨ x3 ∨ ¬x2 ∨ x1 ∨ ¬x0)

    # (x4 ∨ x3 ∨ ¬x2 ∨ ¬x1 ∨ x0)

    # ( x4 ∨ x3 ∨ ¬x2 ∨ ¬x1 ∨ ¬x0)

    # (x4 ∨ ¬x3 ∨ x2 ∨ x1 ∨ ¬x0)

    # ( x4 ∨ ¬x3 ∨ x2 ∨ ¬x1 ∨ x0)

    # (x4 ∨ ¬x3 ∨ x2 ∨ ¬x1 ∨ ¬x0)

    # (x4 ∨ ¬x3 ∨ ¬x2 ∨ x1 ∨ x0)

    # (x4 ∨ ¬x3 ∨ ¬x2 ∨ x1 ∨ ¬x0)

    # (x4 ∨ ¬x3 ∨ ¬x2 ∨ ¬x1 ∨ x0)

    # (x4 ∨ ¬x3 ∨ ¬x2 ∨ ¬x1 ∨ ¬x0)

    # (¬x4 ∨ x3 ∨ x2 ∨ x1 ∨ ¬x0)

    # (¬x4 ∨ x3 ∨ x2 ∨ ¬x1 ∨ x0)

    # (¬x4 ∨ x3 ∨ x2 ∨ ¬x1 ∨ ¬x0)

    # (¬x4 ∨ x3 ∨ ¬x2 ∨ x1 ∨ x0)

    # (¬x4 ∨ x3 ∨ ¬x2 ∨ x1 ∨ ¬x0)

    # (¬x4 ∨ x3 ∨ ¬x2 ∨ ¬x1 ∨ x0)

    # (¬x4 ∨ x3 ∨ ¬x2 ∨ ¬x1 ∨ ¬x0)

    # (¬x4 ∨ ¬x3 ∨ x2 ∨ x1 ∨ x0)

    # (¬x4 ∨ ¬x3 ∨ x2 ∨ x1 ∨ ¬x0)

    # (¬x4 ∨ ¬x3 ∨ x2 ∨ ¬x1 ∨ x0)

    # (¬x4 ∨ ¬x3 ∨ x2 ∨ ¬x1 ∨ ¬x0)

    # (¬x4 ∨ ¬x3 ∨ ¬x2 ∨ x1 ∨ x0)

    # (¬x4 ∨ ¬x3 ∨ ¬x2 ∨ x1 ∨ ¬x0)

    # (¬x4 ∨ ¬x3 ∨ ¬x2 ∨ ¬x1 ∨ x0)

    # (¬x4 ∨ ¬x3 ∨ ¬x2 ∨ ¬x1 ∨ ¬x0)

    return num_variables, answer, {}


def xor_clauses(clauses, num_variables):
    current_var = num_variables
    answer = []
    mapping = dict()
    for clause in clauses:
        current_var, new_clauses, new_mapping = xor_clause(current_var, clause)
        answer.extend(new_clauses)
        mapping.update(new_mapping)
    return answer, current_var, mapping


def resolve_xor_clauses(answer, mapping):
    for (key, value) in mapping.items():
        var1 = bool(answer[abs(value[0])])
        var2 = bool(answer[abs(value[1])])
        answer[key] = int(var1 or var2)


def dump_clauses_to_cnf_file(output_file, clauses, num_variables, comments: List[str]=None):
    if comments is None:
        comments = []
    file = open(output_file, "w")
    for comment in comments:
        file.write(f"c {comment}\n")
    file.write(f"p cnf {num_variables} {len(clauses)}\n")

    for clause in clauses:
        line = ""
        for var in clause:
            if len(line) > 0:
                line += " "
            line += f"{var}"
        line += " 0\n"
        file.write(line)
    file.close()


def unicorn_file_parser(path) -> Tuple[BinaryQuadraticModel, int]:
    bqm = BinaryQuadraticModel(Vartype.BINARY)
    estimated_num_variables = None
    file = open(path)

    current_section = 0
    for line in file.readlines():
        if line == "\n":
            current_section += 1
        else:
            elements = line.split(" ")
            if current_section == 0:
                estimated_num_variables = int(elements[0])
                bqm.offset = float(elements[1])
            elif current_section == 1:
                pass
            elif current_section == 2:
                pass
            elif current_section == 3:
                qubit_id = int(elements[0])
                bias = float(elements[1])
                bqm.add_variable(qubit_id, bias)
            elif current_section == 4:
                qubit_id1 = int(elements[0])
                qubit_id2 = int(elements[1])
                bias = float(elements[2])
                bqm.add_interaction(qubit_id1, qubit_id2, bias)
            else:
                raise Exception("this block of code should be unreachable")

    file.close()
    return bqm, estimated_num_variables


def get_vertices_degrees(bqm):
    answer = []
    for (node, neighbours) in bqm.adj.items():
        answer.append(len(neighbours))
    return answer


def get_bqm_statistics(bqm):
    max_quadratic = max(bqm.quadratic.values())
    max_linear = max(bqm.linear.values())

    vertices_degrees = get_vertices_degrees(bqm)
    avg_degrees = round(float(sum(vertices_degrees)) / float(len(vertices_degrees)), 2)
    variance_degrees = round(variance(vertices_degrees), 2)

    return {
        'max_linear': max_linear,
        'max_quadratic': max_quadratic,
        'max_conn': max(vertices_degrees),
        'variance_conn': variance_degrees,
        'avg_conn': avg_degrees,
        'variables': len(bqm.variables)
    }
