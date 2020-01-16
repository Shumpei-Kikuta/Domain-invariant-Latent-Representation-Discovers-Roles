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
python src/main.py --input graph/10_5_double_barbell.edgelist --num-walks 20 --walk-length 80 --window-size 5 --dimensions 128 --OPT1 True --OPT2 True --OPT3 True --until-layer 6
```

where each arguments are compatible to the paper.   
Then, you will get the embedding in `emb` directory. 

In order to use the embedding for domain adversarial learning, you should move the embedding to the `../adversarial_learning/emb`.

## Adversarial Learning
We use adversarial learning for make both source and target representations domain-invariant.   

We need three kinds of files:
- emb: includes both embedding from struc2vec
- txt: includes label data of each domain data. At least, you need the labels of the source domain.
The format should be as below:
```
node label
```
where label should be related to the role of each node.
- component: specifies which domain a node comes from.
The format should be as below:
```
node component_label
```
where component label should be diffrenet between domains(ex. train -> 0, test -> 1).

In order to implement domain adersarial learning, you need to prepare for the formatting as the following command:

```
python src/translator.py graph/labels-large10-barbell.txt emb/10_5_double_barbell.emb \
                         graph/10_5_double_barbell.component --target graph/labels-small5-barbell.txt 
```

Then, you can execute domain adversarial learning as follows:

```
python src/train.py --is_target_label --param_dir barbell --epoch 1000 \
                    --lambda_ 10 --r_lr 0.0001 --d_lr 0.0001 --suffix barbell
```

You can know the accuracy from the stdout log and parameters under `param_dir`.

# Reference
[1] Ribeiro, Leonardo FR, Pedro HP Saverese, and Daniel R. Figueiredo. "struc2vec: Learning node representations from structural identity." Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2017.
