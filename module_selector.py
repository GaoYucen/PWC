import random

import model_set
import shortest_path
import param

# ! each model is iterated here

def module_selector(model_set, snodes, enodes, routes):
  index = -1
  max_acc = float("-inf")
  
  print("The accuracy of each model is ")
  for i in range(len(model_set)):
    model_set[i].iterate()
    temp_acc = model_set[i].acc_accumulate(snodes, enodes, routes)
    print(i, "th model:", temp_acc)
    temp_acc+=random.expovariate(param.beta)
    if temp_acc > max_acc:
      max_acc = temp_acc
      index = i
  return index
    
