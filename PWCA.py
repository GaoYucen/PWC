#!/home/zzhyyds/anaconda3/bin/python

import numpy as np
import random
import math
import copy
import networkx as nx
import sys
import matplotlib

import model_set
import simulation
import shortest_path
import module_selector
import chasing
import param


real_graph = copy.deepcopy(simulation.read_map())

print(len(real_graph))
G_real = shortest_path.generate_nx_graph(real_graph)


models_set = []

for i in range(len(param.scale_set)):
  temp_model = model_set.Baseline_model(i)
  temp_model.read_graph(real_graph)
  models_set.append(temp_model)

print("model set is initialized")

# ! generate random orders / read from files
if len(sys.argv) > 1:
  if(sys.argv[1]=="-g"):
    simulation.generete_nodes(len(real_graph))
  print("finish orders generation")
  
snodes_l = simulation.read_snodes()
enodes_l = simulation.read_enodes()

# calculate real routes

routes_l = []
for i in range(param.T):
  # print("time slot", i, "finishes calculating")
  routes_l.append(shortest_path.cal_real_routes(G_real, snodes_l[i], enodes_l[i]))




print("finish calculating routes")
print("-------------------------------------------------------------------------------")
# iterate
model_index_set = []
chosen_graph_set = []
acc_set = []

for i in range(param.T):
  print("-------------------------------------------------------------------------------")
  print("The", i, "th iteration starts")
  model_index = module_selector.module_selector(models_set,snodes_l[i],enodes_l[i], routes_l[i])
  model_index_set.append(model_index)
  chosen_graph_set.append(models_set[model_index].generated_graphs[-1])
  temp_acc = chasing.chasing(chosen_graph_set, snodes_l[i], enodes_l[i], routes_l[i])
  acc_set.append(temp_acc)
  print("the",i,"th iteration is finished.", model_index, "th model is chosen.", "the accuracy is",temp_acc)
  print("-------------------------------------------------------------------------------")
print("the indexes of model chosen are: ", end="")
print(model_index_set)
print("the accuracy list is: ",end="")
print(acc_set)
print("accumulation of real acc is", sum(acc_set))

print("accumulation of models' acc are:")
for i in range(len(models_set)):
  print(i, "th model:", models_set[i].acc)
