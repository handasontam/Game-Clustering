# Alpha Beta Communities and Marginal-Increase-Clustering
Marginal Increase Clustering  
[Dr. CHAN Chung](chung.chan@cityu.edu.hk)  
[Ali Al-Bashabsheh](entropyali@gmail.com)  
[Handason Tam](handasontam@gmail.com)  
And big thanks to 
[Zhao Chao](yocopy@outlook.com)  
And thanks also to the implementation of Incremental Breadth-First Search   
"Maximum Flows By Incremental Breadth-First Search" A.V. Goldberg, S.Hed, H. Kaplan, R.E. Tarjan, and R.F. Werneck


# Run Experiment
- install anaconda/miniconda if you haven't: https://conda.io/docs/user-guide/install/index.html
```bash
$ git clone https://github.com/handasontam/Marginal-Increase-Clustering.git
$ conda create --name abcommunities python=3.5  # create virtual conda environment
$ source activate abcommunities
$ pip install -r requirements.txt
$ python run_experiment.py --cpu 2 --data data/example_graph.txt --output /tmp --undirected --unweighted --beta 0.5  # example
```

# Usage
``` bash
$ python train_marginal_increase_clustering.py -h                                                                                    
```

```
usage: train_marginal_increase_clustering.py [-h] --cpu CPU --data DATA
                                             --output OUTPUT [--directed]
                                             [--undirected] [--weighted]
                                             [--unweighted] --beta BETA

finding the alpha-beta dominant communities

optional arguments:
  -h, --help       show this help message and exit
  --cpu CPU        number of cpu core to run the algorithm in parallel
  --data DATA      the file path of the graph data
  --output OUTPUT  the output directory
  --directed
  --undirected
  --weighted
  --unweighted
  --beta BETA      the beta value to use, range: [0,1]
```

# Example Usage
```
# data/example_graph.txt
1 2 1
3 4 1
4 5 1
5 3 1
```

```bash
$ python run_experiment.py --cpu 2 --data data/example_graph.txt --output /tmp --undirected --weighted --beta 0.5
```
