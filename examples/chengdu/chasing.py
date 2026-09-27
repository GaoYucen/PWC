import copy

import model_set
import shortest_path

def chasing(graphs_l, snodes, enodes, routes):
  current_time = len(graphs_l)
  td_graph = shortest_path.time_decay_graph(current_time, graphs_l)
  td_acc = shortest_path.acc_cal(td_graph, snodes, enodes, routes)
  
  return td_acc
  # td_acc = shortest_path.acc_cal(
  #     td_graph, snodes_l[i], enodes_l[i], routes_l[i])
  # td_model_graph = shortest_path.time_decay_graph(
  #     i, baseline_model.generated_graphs)
  # td_model_acc = shortest_path.acc_cal(
  #     td_model_graph, snodes_l[i], enodes_l[i], routes_l[i])
  # acc.append(td_acc)
  # acc_model.append(td_model_acc)
