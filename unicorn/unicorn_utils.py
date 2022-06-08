import sys
from greedy import SteepestDescentSolver
import pandas as pd
import numpy as np
from statistics import variance

sys.path.append("../")
from embedding_utils import *
from utils import *


def get_vertices_degrees(bqm):
    answer = []
    for (node, neighbours) in bqm.adj.items():
        answer.append(len(neighbours))
    return answer


def get_file_bqm(file_name, prefix):
    if prefix is None:
        bqm, estimated_num_variables = unicorn_file_parser(f"./sat_ratio/{file_name}.unicorn")
    else:
        bqm, estimated_num_variables = unicorn_file_parser(f"./{prefix}_experiments/qubos/{prefix}_{file_name}.unicorn")
    vertices_degrees = get_vertices_degrees(bqm)
    avg_degrees = round(float(sum(vertices_degrees)) / float(len(vertices_degrees)), 2)
    variance_degrees = round(variance(vertices_degrees), 2)

    if prefix is None:
        num_variables, num_clauses, variables, clauses = load_cnf(f"./sat_ratio/{file_name}.cnf")
        print(f"file: sat_ratio/{file_name}")
    else:
        num_variables, num_clauses, variables, clauses = load_cnf(
            f"./{prefix}_experiments/cnfs/{prefix}_{file_name}.cnf")

        print(f"file: {prefix}_experiments/{file_name}")
    print(
        f"cnf vars, clauses, clauses_to_var = {num_variables}, {num_clauses}, {round(num_clauses / num_variables, 2)}")
    print(f"qubo vars {len(bqm.variables)}")
    print(f"conn avg, variance, max = {avg_degrees}, {variance_degrees}, {max(vertices_degrees)}")

    return bqm


def get_input_value(sample, one_input):
    answer = 0
    for (index, element) in enumerate(one_input):
        answer += (sample[element]) * (2 ** index)
    return int(answer)


def get_input_values(sample, input_ids):
    answer = []

    for i in input_ids:
        answer.append(get_input_value(sample, i))
    return ','.join([str(x) for x in answer])


def get_results_dataframe(file, input_ids, reads=100, bottom=0.25, top=5, is_pegasus=True, is_chimera=True,
                          is_local_search=True, random_seed_pegasus=None, random_seed_chimera=None,
                          pegasus_qpu=None, chimera_qpu=None, experiments_folder=None):
    solver_greedy = SteepestDescentSolver()
    bqm = get_file_bqm(file, experiments_folder)

    chain_strengths = []
    energies = []
    input_vals = []
    file_name = []
    sampleset_types = []
    qpus = []
    if is_pegasus:
        if random_seed_pegasus:
            embedding_1 = get_embedding(bqm, pegasus_qpu, random_seed=random_seed_pegasus)
        else:
            embedding_1, random_seed = find_best_embedding(bqm, pegasus_qpu)
        sampler = FixedEmbeddingComposite(pegasus_qpu, embedding_1)
        for chain_strength in list(np.arange(bottom, top + 0.25, 0.25)):

            sampleset = sampler.sample(bqm, num_reads=reads, chain_strength=chain_strength, auto_scale=True,
                                       answer_mode='raw')
            assert (len(sampleset) == reads)
            print("advantage:", chain_strength, "raw", sampleset.first.energy)
            for sample in sampleset:
                file_name.append(file)
                chain_strengths.append(chain_strength)
                energies.append(bqm.energy(sample))
                sampleset_types.append("raw")
                qpus.append("Advantage4.1")
                input_vals.append(get_input_values(sample, input_ids))

            if is_local_search:
                sampleset_pp = solver_greedy.sample(bqm, initial_states=sampleset)
                print(chain_strength, "pp_local_search", sampleset.first.energy)
                for sample in sampleset_pp:
                    file_name.append(file)
                    chain_strengths.append(chain_strength)
                    energies.append(bqm.energy(sample))
                    input_vals.append(get_input_values(sample, input_ids))
                    qpus.append("Advantage4.1")
                    sampleset_types.append("pp_local_search")
                print("************")
    if is_chimera:
        if random_seed_chimera is None:
            embedding_2, random_seed = find_best_embedding(bqm, chimera_qpu)
        else:
            embedding_2 = get_embedding(bqm, chimera_qpu, random_seed=random_seed_chimera)
        sampler = FixedEmbeddingComposite(chimera_qpu, embedding_2)
        for chain_strength in list(np.arange(bottom, top + 0.25, 0.25)):
            print("reads:", reads)
            sampleset = sampler.sample(bqm, num_reads=reads, chain_strength=chain_strength, auto_scale=True,
                                       answer_mode='raw')
            assert (len(sampleset) == reads)
            print('chimera:', chain_strength, "raw", sampleset.first.energy)
            for sample in sampleset:
                file_name.append(file)
                chain_strengths.append(chain_strength)
                energies.append(bqm.energy(sample))
                sampleset_types.append("raw")
                qpus.append("DW_2000Q_6")
                input_vals.append(get_input_values(sample, input_ids))
            if is_local_search:
                sampleset_pp = solver_greedy.sample(bqm, initial_states=sampleset)
                print(chain_strength, "pp_local_search", sampleset.first.energy)
                for sample in sampleset_pp:
                    file_name.append(file)
                    chain_strengths.append(chain_strength)
                    energies.append(bqm.energy(sample))
                    input_vals.append(get_input_values(sample, input_ids))
                    qpus.append("DW_2000Q_6")
                    sampleset_types.append("pp_local_search")
                print("************")

    return pd.DataFrame({
        'file': file_name,
        'type': sampleset_types,
        'chain_strength': chain_strengths,
        'energy': energies,
        'input_values': input_vals,
        'qpu': qpus
    })


def get_percentage_ground_states(df, chain_strength):
    temp_df = df[df.chain_strength == chain_strength]
    total_samples = temp_df.shape[0]
    print("total_samples:", total_samples)
    temp_df = temp_df[temp_df.energy == 0]
    return round(temp_df.shape[0]/total_samples,2)


def get_best_chain_strength(filename, qpu='DW_2000Q_6'):
    df = pd.read_csv(filename)
    df = df[df.type == 'raw']
    df = df[df.qpu == qpu]
    max_percentage = None

    chain_strengths = df.chain_strength.unique()

    # find_max
    for cs in chain_strengths:
        percentage = get_percentage_ground_states(df, cs)
        if max_percentage == None:
            max_percentage = percentage
        else:
            max_percentage = max(max_percentage, percentage)

    best_chain_strengths = []

    for cs in chain_strengths:
        percentage = get_percentage_ground_states(df, cs)
        if percentage == max_percentage:
            best_chain_strengths.append(cs)

    return best_chain_strengths, max_percentage
