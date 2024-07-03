import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from queue import Queue
from collections import defaultdict
from random import choice

path_ids = "Data/itwiki-2013.ids"
page_to_id = {}
id_to_page = {}
with open(path_ids, 'r', encoding='utf-8') as reader_1:
    for i, line in enumerate(reader_1):
        line = line.strip()
        page_to_id[line] = i
        id_to_page[i]=line

print(id_to_page.get(261576))
print(page_to_id.get("Città dell'India"))

path_arcs = "Data/itwiki-2013.arcs"
arcs_df_prova = pd.read_csv(path_arcs,sep=" ", header=None, names=['v1','v2'])


def create_graph(df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()

    for index, line in df.iterrows():
        v1 = int(line['v1'])
        v2 = int(line['v2'])
        G.add_edge(v1, v2)
    return G

g_prova = create_graph(arcs_df_prova)
print(g_prova)

def outDegree_distribution (g:nx.DiGraph) -> dict:
    neighbors_dict ={node: list(g.successors(node)) for node in g.nodes()} # da nodo a lista vicini
    distribution_dict = {} # da grado a array di 2 elementi: in posizione 0, nodi con quel grado; in posizione 1, conteggio nodi con quel grado
    for node in neighbors_dict.keys():
        degree_node = len(neighbors_dict[node])
        if degree_node in distribution_dict:
            distribution_dict[degree_node][0].append(node)
            distribution_dict[degree_node][1]+=1
        else:
            distribution_dict[degree_node] = [[node],1]
    return distribution_dict
g_outDegree = outDegree_distribution(g_prova)

degrees = list(g_outDegree.keys())
frequencies = [g_outDegree[degree][1] for degree in degrees]

plt.bar(degrees, frequencies)
plt.xlabel('Out-degree')
plt.ylabel('Frequency')
plt.xlim([0,200])
plt.ylim([0,max(frequencies)])
plt.title('Out-degree Distribution')
plt.show()

max_degree = max(degrees)
top_10 = list()

while len(top_10)<10:
    if max_degree in g_outDegree.keys():
        nodes = list(g_outDegree[max_degree][0])
        top_10.extend(nodes)
    max_degree = max_degree - 1
top10 = top_10[:10]
top10_pages = []
for id in top10:
    top10_pages.append(id_to_page[id])
top10_pages = pd.Series(top10_pages, index=range(1,11))
print(top10_pages)

def get_largest_cc_undirected(G:nx.DiGraph)->nx.Graph:
    G = nx.to_undirected(G)
    return G.subgraph(
    sorted(nx.connected_components(G), key = len, reverse=True)[0]
    ).copy()

LCC = get_largest_cc_undirected(g_prova)

def customBFS(LCC, startNode):
    visited = {}
    queue = Queue()
    queue.put(startNode)
    visited[startNode] = 0
    while not queue.empty():
        currentNode = queue.get()
        for nextNode in LCC.neighbors(currentNode):
            if nextNode not in visited:
                queue.put(nextNode)
                visited[nextNode]=visited[currentNode]+1
    B_u = defaultdict(list)
    for key, value in visited.items():
        B_u[value].append(key)
    return B_u

def computeDiameter(LCC, Bu):
    i = lb = max(Bu)
    ub = 2*lb
    while ub > lb:
        eccDict = nx.eccentricity(LCC, Bu[i])
        Bi = max(eccDict.values())
        maxVal = max(Bi,lb)
        if maxVal > 2*(i - 1):
            return print("diametro: ", maxVal)
        else:
            lb = maxVal
            ub = 2*(i - 1)
        i = i - 1
    return print("Diametro iFub: ", lb)

startNode = max(LCC.degree,key=lambda x: x[1])[0]
B_u = customBFS(LCC, startNode)
computeDiameter(LCC,B_u)


def create_graph_without_disambigua(df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()

    for index, line in df.iterrows():
        v1 = int(line['v1'])
        v2 = int(line['v2'])

        if "disambigua" not in id_to_page[v1] and "disambigua" not in id_to_page[v2]:
            G.add_edge(v1, v2)
    return G


G_without_disambigua_prova = create_graph_without_disambigua(arcs_df_prova)

LCC_dis = get_largest_cc_undirected(G_without_disambigua_prova)
startNode_dis = max(LCC_dis.degree, key=lambda x: x[1])[0]
B_u_dis = customBFS(LCC_dis, startNode_dis)
computeDiameter(LCC_dis, B_u_dis)

U_g = nx.to_undirected(g_prova)


def Bron_Kerbosch(G, R, P, X):
    if not P and not X:
        return R
    for v in list(P):
        myclique = Bron_Kerbosch(
            G,
            R.union({v}),
            P.intersection(G.neighbors(v)),
            X.intersection(G.neighbors(v))
        )
        if myclique and len(myclique) >= 3:
            return myclique
        P.remove(v)
        X.add(v)
    return None


def find_a_maximal_clique(G: nx.Graph()):
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        start_node = choice(nodes)
        R = {start_node}
        P = set(G.neighbors(start_node))
        X = set()
        myclique = Bron_Kerbosch(G, R, P, X)
        if myclique and len(myclique) >= 3:
            myclique_pages = set()
            for id in myclique:
                myclique_pages.add(id_to_page[id])
            return print("Una clique massimale di ordine 3 o superiore è: ", myclique_pages)
    return "Nessuna clique di ordine 3 trovata"


find_a_maximal_clique(U_g)


def find_n_maximal_cliques(G, n):
    nodes = list(G.nodes())
    maximal_cliques = []
    iter = 0
    while len(maximal_cliques) < n:
        iter += 1
        start_node = choice(nodes)
        R = {start_node}
        P = set(G.neighbors(start_node))
        X = set()
        myclique = Bron_Kerbosch(G, R, P, X)
        if myclique and len(myclique) >= 3 and myclique:
            duplicate = False
            for old_clique in maximal_cliques:
                if myclique == old_clique:
                    duplicate = True
            if not duplicate:
                maximal_cliques.append(myclique)
                myclique_pages = set()
                for id in myclique:
                    myclique_pages.add(id_to_page[id])
                print("Clique massimale ", iter, ": ", myclique_pages)

    return maximal_cliques


find_n_maximal_cliques(U_g, 2)