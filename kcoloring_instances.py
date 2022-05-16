from typing import Dict

from networkx import Graph
from cnfgen.cnf import CNF
from cnfgen.families.coloring import GraphColoringFormula
from dwave.embedding.pegasus import find_clique_embedding
from embedding_utils import get_qpu
from utils import cnf_to_bqm


pegasus_graph = get_qpu().to_networkx_graph()

def get_clique_graph(n) -> Graph:
    '''

    :param n: number of nodes in the graph
    :return: networkx graph
    '''
    edgelist = []
    for i in range(1,n+1):
        for j in range(i+1,n+1):
            edgelist.append((i,j))

    g = Graph(edgelist)
    return g


def get_kcoloring_cnf_from_graph(g: Graph, colors: int = None):
    if colors is None:
        colors = len(g.nodes)
    return GraphColoringFormula(g, colors)


def get_k_coloring_cnf(n, colors: int = None):
    g = get_clique_graph(n)
    return get_kcoloring_cnf_from_graph(g, colors)


def get_mapping_variables(cnf: CNF) -> Dict[str, int]:
    answer: Dict[str, int] = dict()
    c = 1
    for v in cnf.variables():
        answer[v] = c
        c+=1
    return answer

def get_cnf_objects(cnf: CNF):
    mapping_variables = get_mapping_variables(cnf)
    variables = set()
    for val in mapping_variables.values():
        assert(val not in variables)
        variables.add(val)

    clauses = []
    for clause in cnf.clauses():
        c = []

        for (b, var) in clause:
            v = mapping_variables[var]
            if not b:
                v *= -1
            c.append(v)
        clauses.append(c)

    return variables, clauses, mapping_variables


def get_clique_embedding(n: int):
    return find_clique_embedding(n, 16)

def coloring_bqm(n, colors: int = None):
    embedding = find_clique_embedding(n)
    cnf = get_k_coloring_cnf(n, colors)
    variables, clauses, mapping_variables = get_cnf_objects(cnf)
    bqm = cnf_to_bqm(variables, clauses)
    return bqm, embedding, variables, clauses, mapping_variables




