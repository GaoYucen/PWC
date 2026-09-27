import random
import numpy as np
import copy

import shortest_path
import param


# Define baseline methods
class Baseline_model:
  # base_graph is the real graph, which we will use to generate random weight
  def __init__(self, s_index):
    self.s_index = s_index
    self.scale = param.scale_set[s_index]
    # self.L = L
    self.base_graph = {}
    self.generated_graphs = []
    self.acc = 0
    self.w = 1/len(param.scale_set)

  # graph is a dict of dict, representing the adjacency list
  def read_graph(self, graph):
    self.base_graph = copy.deepcopy(graph)

  # generate the graph for next iteration
  def iterate(self):
    new_graph = {}
    scale = self.scale
    for key, value in self.base_graph.items():
      a_node_set = {}
      for adjacent_node, weight in value.items():
        a_node_set[adjacent_node] = weight + scale * random.random()
      new_graph[key] = copy.deepcopy(a_node_set) # TODO: do we need to deep copy again?
    self.generated_graphs.append(new_graph)
        
  def acc_accumulate(self, snodes, enodes, routes):
    td_graph = shortest_path.time_decay_graph(len(self.generated_graphs), self.generated_graphs)
    new_acc = shortest_path.acc_cal(td_graph, snodes, enodes, routes)
    self.acc+=new_acc
    return new_acc

