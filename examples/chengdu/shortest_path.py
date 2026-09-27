import networkx as nx
import copy
import math

import param


def generate_nx_graph(graph):
  G = nx.DiGraph()
  for key, value in graph.items():
    for node, weight in value.items():
      G.add_edge(key,node,weight=weight)
  return G

def shortest_path(G, snode, enode):
  return nx.dijkstra_path(G, source=snode, target=enode, weight="weight")


def cal_real_routes(G, snodes, enodes):
  routes = []
  for i in range(len(snodes)):
    temp = shortest_path(G, snodes[i], enodes[i])
    routes.append(temp)
  return routes


def acc_cal(graph, snodes, enodes, routes):
  G = generate_nx_graph(graph)
  correct_num = 0
  for i in range(len(routes)):
    temp_route = shortest_path(G, snodes[i], enodes[i])
    if (temp_route==routes[i]):
      correct_num+=1
  return correct_num/(len(routes))

def time_decay_graph(t, graphs_l):
    starting_index = t - param.L
    if (starting_index < 0):
      starting_index = 0
    rst = copy.deepcopy(graphs_l[starting_index])
    for i in range(starting_index + 1, t):
      for key, value in graphs_l[i].items():
        for node, weight in value.items():
          rst[key][node] = rst[key][node]/math.e + weight
    return rst
