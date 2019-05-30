# MTL-suite
Suite of multitask learning methods and a framework for easy experimentation with machine learning methods.


Title: Multi-Task Learning Software Suite (MTL4C), Version: 1.0

## Repository content
This repository contains a general purpose multitask learning (MTL) software suite implemented in Python. Classification and regression methods are included. In the MTL setup, a set of related tasks (regression or classification, for example) need to be solved and all models are learned jointly through a shared representation that allows for information transference accross related tasks. Notice that in MTL there is still a model associated with each task, the crucial point is the joint learning.

Two baseline methods are also implemented, namely: *single task learning* (STL), where a model is trained for each task independently; and *pooled model*, where a single model is trained for all tasks. The machine learning methods available in this software are located the folder named `methods`. Linear and non-linear models are available.


## Quick start

Two demo files are available in the `experiments` folder: `demo_regression.py` and `demo_classification.py`.
One can build up on these files to run new experiments.

> python demo_classification.py

## Authors
- André R. Gonçalves (goncalves1@llnl.gov)
- David P. Widemann (widemann1@llnl.gov)
- Braden C. Soper (soper3@llnl.gov)
- Pryiadip Ray (ray34@llnl.gov)
- Jan Nygard (Jan.Nygard@kreftregisteret.no)
- Mari Nygard (Mari.Nygard@kreftregisteret.no)
- Ana Paula Sales (deoliveirasa1@llnl.gov) 

## References

Gonçalves, A.R., Von Zuben, F. J, and Banerjee, A. "Multi-task sparse structure learning with gaussian copula models." The Journal of Machine Learning Research 17.1 (2016): 1205-1234. [paper](http://jmlr.org/papers/v17/15-215.html)

---
LLNL-CODE-773785
