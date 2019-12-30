# Discription
This repository provides the implementation for `Domain-invariant-Latent-Representation-Discovers-Roles`.
Here is the discription of the paper:

```
Domain-invariant-Latent-Representation-Discovers-Roles
S. Kikuta, F. Toriumi, M.Nishiguchi, T.Fukuma, T.Nishida, and S. Usui.
The 8th International Conference on Complex Networks and their Applications
```

# Components
- struc2vec
We use `struc2vec`[1] for embedding nodes in the source and the target networks into the same dimensional space.   
Structurally related latent representations are to be generated.   

- Adversarial Learning
We use domain adversarial learning to make both latent representations domain-invariant, and discriminative for the source domain.

# How to Use
## struc2vec
The implementation is largely similar to [original repository](https://github.com/leoribeiro/struc2vec).    
The difference is that this implementation runs on python3 (whereas original version is on python2).

First, you should put the edgelist to the `graph` directory.

Go under `struc2vec` directory, and execute the following command to embed nodes into k-dimensional spaces:

```
python src/main.py --input graph/karate-mirrored-edges.edgelist --num-walks 20 --walk-length 80 --window-size 5 --dimensions 128 --OPT1 True --OPT2 True --OPT3 True --until-layer 6
```

where each arguments are compatible to the paper.   
Then, you will get the embedding in `emb` directory. 

In order to use the embedding for domain adversarial learning, you should move the embedding to the `../adversarial_learning/emb`.




# Reference
[1] Ribeiro, Leonardo FR, Pedro HP Saverese, and Daniel R. Figueiredo. "struc2vec: Learning node representations from structural identity." Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2017.
