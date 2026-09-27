import copy
import sys

import chasing
import model_set
import module_selector
import read_graph
import shortest_path

real_graph = copy.deepcopy(read_graph.read_map())

print(len(real_graph))
G_real = shortest_path.generate_nx_graph(real_graph)

models_set = []
for i in range(len(model_set.param.scale_set)):
    temp_model = model_set.Baseline_model(i)
    temp_model.read_graph(real_graph)
    models_set.append(temp_model)

print("model set is initialized")

if len(sys.argv) > 1 and sys.argv[1] == "-g":
    read_graph.generete_nodes(len(real_graph))
    print("finish orders generation")

snodes_l = read_graph.read_snodes()
enodes_l = read_graph.read_enodes()

routes_l = []
for i in range(model_set.param.T):
    routes_l.append(shortest_path.cal_real_routes(G_real, snodes_l[i], enodes_l[i]))

print("finish calculating routes")

model_index_set = []
chosen_graph_set = []
acc_set = []

for i in range(model_set.param.T):
    model_index = module_selector.module_selector(
        models_set, snodes_l[i], enodes_l[i], routes_l[i]
    )
    model_index_set.append(model_index)
    chosen_graph_set.append(models_set[model_index].generated_graphs[-1])
    temp_acc = chasing.chasing(
        chosen_graph_set, snodes_l[i], enodes_l[i], routes_l[i]
    )
    acc_set.append(temp_acc)

print("the indexes of model chosen are:", model_index_set)
print("the accuracy list is:", acc_set)
print("accumulation of real acc is", sum(acc_set))
print("accumulation of models' acc are:")
for i, model in enumerate(models_set):
    print(i, "th model:", model.acc)
