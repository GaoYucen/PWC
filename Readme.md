# PWCA

## Structure

```mermaid
graph LR;
  PWCA.py-->module_selector.py;
  module_selector.py-->model_set.py;
  PWCA.py-->shortest_path.py;
  PWCA.py-->simulation.py;
  simulation.py-->files;
  files-->simulation.py;
  PWCA.py-->param.py;
  module_selector.py-->param.py;
  model_set.py-->param.py;
  shortest_path.py-->param.py;
  simulation.py-->param.py;
```

## Parameters

1. Need to be modified, default compiler in first line of every python file
2. `-g`: generate new nodes and write them to files
   * For example, `./PWCA.py -g`
   * `./PWCA.py` will read orders from files
3. *Place to change input files:* `param.py`
