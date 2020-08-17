# Tensorflow on `Steroids` : 

## This repo contains the following:  

1. Replicated `OneCycleLR` learning rate scheduler from `PyTorch` into a `tf.keras` `callback` .   
   - [Original implementation by pytorch](https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CyclicLR)
   - [My Version](https://github.com/benihime91/tensorflow-keras-nbs/blob/master/one_cycle.py).

![one_cycle_lr_plot](vis/one_cycle_plots.png)  


2. `Learning Rate Finder` for `tf.keras`
   - Main idea taken from the implementation of `lr_find` by [Fast.ai](https://docs.fast.ai/basic_train.html#lr_find)
   - Ideas taken from [How Do You Find A Good Learning Rate](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html).
   - [My Version](https://github.com/benihime91/tensorflow-keras-nbs/blob/master/lr_find.py). 

> To check the usage of the above mentioned functionalities check this [notebook](https://github.com/benihime91/tensorflow-keras-nbs/blob/master/one_cycle_%26_lr_finder_tf.ipynb).  

![Lr_finder Plot](vis/lr_finder_plot_1.png)
![Lr_finder Plot](vis/lr_finder_plot_2.png)

3. Example on how to use `Albumentations` for `image augmentation` with `Tensorflow's` `tf.data` API  
> Check this [notebook](https://github.com/benihime91/tensorflow-keras-nbs/blob/master/albumentations_with_tensorflow.ipynb)   
There are some issues though with this implementation. Please check [this](https://github.com/albumentations-team/albumentations/issues/669#issuecomment-664422245).
Tried my best to showcase the examples where this is gonna work and not work.
