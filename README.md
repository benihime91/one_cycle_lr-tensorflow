# tensorflow-on-steroids : 

## This repo contains the following:  

1. Replicated `OneCycleLR` learning rate scheduler from `PyTorch` into a `tf.keras` `callback` .   
   - [Original implementation by pytorch](https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CyclicLR)
   - [My Version](https://github.com/benihime91/tensorflow-on-steroids/blob/master/one_cycle.py).
   
**Usage:**
```bash
git clone https://github.com/benihime91/tensorflow-on-steroids.git
cd tensorflow-on-steroids
```

**Example :** 
```python

# Import `OneCycleLr`
from one_cycle import OneCycleLr

# Configs
max_lr = 5e-02
epochs = 5

# Istantiate `OneCycleLr`
one_c = OneCycleLr(max_lr=max_lr, steps_per_epoch=len(trn_ds), epochs=epochs)

# Instantiate CallbackList
cbs = [one_c, ...]

# Instantiate Optimizer & loss_fn
optim = keras.optimizers.SGD(momentum=0.9, clipvalue=0.1)
loss_fn = ...

# Compile Model
model.compile(optimizer=optim, loss=loss_fn, metrics=["acc"])

# Fit Model
h = model.fit(trn_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs)

# to plot the learning_rate & momentum(or beta_1) graphs
one_c.plot_lrs_moms()
```
![one_cycle_lr_plot](vis/one_cycle_plots.png)  


2. `Learning Rate Finder` for `tf.keras`
   - Main idea taken from the implementation of `lr_find` by [Fast.ai](https://docs.fast.ai/basic_train.html#lr_find)
   - Ideas taken from [How Do You Find A Good Learning Rate](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html).
   - [My Version](https://github.com/benihime91/tensorflow-on-steroids/blob/master/lr_find.py). 
  
**Example:**
```python
# Import LrFinder
from lr_find import LrFinder

# Instantiate Optimizer & loss_fn 
# [must be instance of tf.keras.Optimizers & tf.keras.Losses]
optimizer = ...
loss_fn = ...

# Instantiate LrFinder
lr_find = LrFinder(model, optimizer, loss_fn)

# Start range_test
lr_find.range_test(trn_ds)

# Plot LrFinder graphs
lr_find.plot_lrs()
```
![Lr_finder Plot](vis/lr_finder_plot_1.png)

```python
# Plot LrFinder graphs
lr_find.plot_lrs(skip_end=0, suggestion=True)
```
![Lr_finder Plot](vis/lr_finder_plot_2.png)

**NB:  
To check the usage of `lr_find` & `OneCycleLR` check this [notebook](https://github.com/benihime91/tensorflow-on-steroids/blob/master/nbs/one_cycle_%26_lr_finder_tf.ipynb).  
This notebook contains an end-to-end example using the `cats-vs-dogs` dataset from `Kaggle`** 

## References & Citations:
```
@misc{smith2015cyclical,
    title={Cyclical Learning Rates for Training Neural Networks},
    author={Leslie N. Smith},
    year={2015},
    eprint={1506.01186},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
```
@misc{howard2018fastai,
  title={fastai},
  author={Howard, Jeremy and others},
  year={2018},
  publisher={GitHub},
  howpublished={\url{https://github.com/fastai/fastai}},
}
```
```
@incollection{NEURIPS2019_9015,
title = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
author = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {8024--8035},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf}
}
```
```
https://github.com/bckenstler/CLR
```
