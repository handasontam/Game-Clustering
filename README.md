# Alpha Beta Clustering and Marginal-Increase-Clustering
Marginal Increase Clustering  
[Dr. CHAN Chung](chung.chan@cityu.edu.hk)  
[Ali Al-Bashabsheh](entropyali@gmail.com)  
[Handason Tam](handasontam@gmail.com)  
And big thanks to 
[Zhao Chao](yocopy@outlook.com)

# Run Experiment
- install anaconda/miniconda if you haven't: https://conda.io/docs/user-guide/install/index.html
```bash
$ git clone https://github.com/handasontam/Marginal-Increase-Clustering.git
$ conda create --name abclustering python=3.5  # create virtual conda environment
$ source activate abclustering
$ pip install -r requirements.txt
$ python train_marginal_increase_clustering.py --cpu 2 --data data/example_graph.txt --output /tmp --undirected --weighted --beta 0.5  # example
```
