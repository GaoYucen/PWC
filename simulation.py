#!/home/zzhyyds/anaconda3/bin/python
import pickle
import numpy as np
import time
import random
import sys
import networkx as nx
import heapq

import param



def read_map():
  graph = {}
  # The file graph.txt, each row has snode enode dist, read them into the map
  with open(param.GRAPH_FILE, "r") as map_file:
    for line in map_file:
      data = line.strip().split(" ")
      snode = int(data[0])
      enode = int(data[1])
      length = float(data[2])
      
      if snode not in graph:
        graph[snode] = {}
      graph[snode][enode] = length
      
      if enode not in graph:
        graph[enode] = {}
  return graph


def generete_nodes(s: int):
    snodes = []
    enodes = []

    PARA = int(s / (param.T / 9))
    num_order = 100

    for i in range(param.T):
        crt_snodes = []
        crt_enodes = []
        for j in range(num_order):
            region = int(i / 10)
            min_value = region * PARA
            if(region==0):
              min_value = 1
            snode = random.randint(min_value, (region + 1) * PARA - 1)
            enode = random.randint(min_value, (region + 1) * PARA - 1)
            crt_snodes.append(snode)
            crt_enodes.append(enode)
        snodes.append(crt_snodes)
        enodes.append(crt_enodes)

    with open(param.SNODE_FILE, "wb") as f:
        pickle.dump(snodes, f)
    with open(param.ENODE_FILE, "wb") as f:
        pickle.dump(enodes, f)


def read_snodes():
  with open(param.SNODE_FILE, "rb") as f:
    mylist = pickle.load(f)
  return mylist

def read_enodes():
  with open(param.ENODE_FILE, "rb") as f:
    mylist = pickle.load(f)
  return mylist
