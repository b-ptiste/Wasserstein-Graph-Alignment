**Authors** : 

- CALLARD Baptiste (MVA): baptiste.callard@ens-paris-saclay.fr 
- TOCQUEC Louis (MVA): louis.tocquec@ens-paris-saclay.fr

# Wasserstein-Graph-Alignment
The project was carried out as part of Jean Feydy's Geometric data analysis course (see. https://www.jeanfeydy.com/Teaching/index.html) in the Master MVA (ENS Paris Saclay).
We studied and re-implemented from scratch the "Wassertein-based graph alignment" paper, whose code is not available in open source.

The idea of this paper is to propose a method for aligning graphs of different sizes using optimal transport methods. Their method, although non-convex, can be optimised using gradient decadence and they have proposed a new derivable Dykstra operator.

After re-implementing the method, we set out to study the cases in which the model behaved optimally, the failure modes, and how to improve the convergence of the model. We also wanted to go further with the experiments and propose simple illustrations to present our experiments and results.
# Installation 

```
git clone https://github.com/b-ptiste/Wasserstein-Graph-Alignment.git
conda create -n wassertein-graph-env python=3.10.13
conda activate wassertein-graph-env
pip install -r requirements.txt
(optional) pip list
```

# Familiarising yourself with the project
- Consult our notebooks
- Read our review-paper

# Some plot illustration the final output of the model

- Convergence and evolution of graph alignments during learning (each image corresponds to a learning stage)

https://github.com/b-ptiste/Wasserstein-Graph-Alignment/assets/75781257/9bdbf616-458c-401a-ac97-cdf3ea7ed0b7

- Experimental results comparing the distance between well-chosen graphs 

<div align="center">
  <img src="https://github.com/b-ptiste/Wasserstein-Graph-Alignment/assets/75781257/74c21fdd-50ee-4748-b0e6-460d28db64af" width="400"/>
  <img src="https://github.com/b-ptiste/Wasserstein-Graph-Alignment/assets/75781257/2b514895-4c97-4c8f-9374-c1b02de66a35" width="400"/>
</div>



# credit
**Paper :** Wasserstein-based graph alignment.
Maretic, Hermina Petric, et al. "Wasserstein-based graph alignment." IEEE Transactions on Signal and Information Processing over Networks 8 (2022): 353-363.

