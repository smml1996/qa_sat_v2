from minorminer import find_embedding
from dwave.system import DWaveSampler
from dimod import BinaryQuadraticModel


def get_pegasus_qpu():
    return DWaveSampler(solver={"name": "Advantage_system4.1"})


def get_embedding(bqm: BinaryQuadraticModel, random_seed: int = 1):
    qpu_pegasus = get_pegasus_qpu()
    return find_embedding(bqm.quadratic.keys(), qpu_pegasus.edgelist, random_seed=random_seed)


def get_chain_lengths(bqm, _embedding):
    lengths = []
    biases = []
    for (key, value) in _embedding.items():
        lengths.append(len(value))
        biases.append(bqm.linear[key])
    return lengths, biases


def count_qubits_used(embedding):
    physical_vars = set()
    for (logic_var, chain) in embedding.items():
        for pv in chain:
            physical_vars.add(pv)
    return len(physical_vars)


def get_chains_with_length(embedding, length):
    answer = []
    for (logic_var, chains) in embedding.items():
        if len(chains) == length:
            answer.append(logic_var)
    return answer


def get_qubits_used(embedding):
    qubits_used = set()
    for (logic_var, chain) in embedding.items():
        for c in chain:
            qubits_used.add(c)

    return qubits_used


def get_qpu():
    return DWaveSampler(solver={"name": "Advantage_system4.1"})


def get_qubit_offset_ranges():
    qpu = get_qpu()
    return qpu.properties['anneal_offset_ranges']


def get_annealing_step():
    qpu = get_qpu()
    return qpu.properties['anneal_offset_ranges']


def get_0_qubits_offsets():
    """

    :return: vector with zeros. The size of the vector is the number of qubits in the qpu.
    """
    qubit_offset_ranges = get_qubit_offset_ranges()
    return [0] * len(qubit_offset_ranges)


def get_min_offset(qpu_offsets, logic_var):
    return qpu_offsets[logic_var][0]


def get_max_offset(qpu_offsets, logic_var):
    return qpu_offsets[logic_var][1]


def advance_annealing_of_unused(embedding, qubit_offsets):
    qubit_offset_ranges = get_qubit_offset_ranges()

    qubits_used = get_qubits_used(embedding)

    for i in range(len(qubit_offset_ranges)):
        if i not in qubits_used:
            max_offset = qubit_offset_ranges[i][1]
            qubit_offsets[i] = max_offset

def get_input_vars(num_variables, original_vars_to_mirrors):
    pass

def set_logic_var_annealing_offsets(embedding, qubit_offsets, num_variables, get_offset, original_vars_to_mirrors=None):
    qpu_offsets = get_qubit_offset_ranges()
    input_vars = get_input_vars(num_variables, original_vars_to_mirrors)
    for (logic_var, chain) in embedding.items():
        if logic_var <= num_variables:
            for c in chain:
                qubit_offsets[c] = get_offset(qpu_offsets, c)

def find_qubit(physical_var, embedding):
    for (logic_var, chain) in embedding.items():
        for c in chain:
            if c == physical_var:
                return logic_var
    return -1

