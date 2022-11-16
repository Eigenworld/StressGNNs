# 		Tired of Over-smoothing? 

# Stress Graph Drawing Is All You Need!

**Binary Stress Function:**

$\text{B}(p)=\sum_{<i,j>\in E}||p_i-p_j||^2+\theta(||p_i-p_j||-1)^2$

where $\theta$ is the balance coefficient. The first item emphasizes the local structure by making the graph layout of connected nodes more compact, while the second item strives to distribute all nodes evenly within a circle.

**Binary Stress Graph Drawing**

<img src="/Users/lixue/Documents/GitHub/StressGNNs/img/B-Stress.png" alt="B-Stress" style="zoom:90%;" />

**Full Stress Function:**

$\text{Stress}(P)= \sum_{\in E}w_{ij}(||P_i-P_j||-d_{ij})^2+\sum_{\in { V \choose 2 }\backslash E}w_{ij}(||P_i-P_j||-d_{ij})^2$

 The first item is called attractive forces, which tend to shorten edges and maintain the compactness of connected nodes. Repulsive forces in the second item keep all nodes $V$ well separated. 

**Full Stress Graph Drawing**:

<img src="/Users/lixue/Documents/GitHub/StressGNNs/img/F-Stress.png" alt="F-Stress" style="zoom:90%;" />







## My Experiment Environments
* [Python = 3.7](https://www.python.org/)
* [Pytorch = 1.5.0](https://pytorch.org)
* [Pytorch_Geometric = 1.5.0](https://pytorch-geometric.readthedocs.io/en/latest/)
* [Cuda = 10.2](https://pytorch.org)
* [GPU-> 'TITAN RTX'](https://pytorch.org)
* Recommend: Use jupyter notebook to see our ipynb file (not in GitHub!!)


## Code Architecture
    .F o l d e r
    ├── img                    # images for readme.md
    ├── data                   # benchmark networks 
    ├── Experiments 👇
    Experiment-Files:
       ├── Github_Linear_Attractive_Models.ipynb                 
       ├── Github_Nonlinear_Attractive_Models.ipynb					
       ├── Github_Repulsive_Models.ipynb
       ├── Github_Virtual_Pivot_Models.ipynb
       ├── Github RWN-DA-DAD Random Test.ipynb
       ├── Github GCN-SAGE Random Feature Test.ipynb
    Other-Files:
       ├── karate_nx.edgelist
       ├── D_A_D.py
       ├── RWN_5_60_cora_300.txt
       ├── DA_5_60_cora_300.txt

