# Faster training on CIFAR10

Demonstration of training a small ResNet on CIFAR10 to 94% test accuracy in the minimal possible time and under 20 epochs.

This code is inspired by [cifar10-fast repository](https://github.com/davidcpage/cifar10-fast) and some of
the current code is adapted from the repository. In his [blog articles](https://www.myrtle.ai/2018/09/24/how_to_train_your_resnet/)
David Page (@davidcpage) explains the choice of the model and the way to optimize the dataflow.

## Project on Polyaxon platform

### Create project
```
polyaxon project create --name=cifar10-faster --description="Faster training on CIFAR10"
```

### Initialize project 
```
polyaxon init cifar10-faster
```

## Reproduce original result

Train `fast-resnet` during 24 epochs using cutout data augmentation, SGD optimizer, piecewise linear scheduling:

```
polyaxon run -u -f plx_configs/fastresnet/xp_original_training.yaml --name=xp_original_training --tags=original
```

## Check other configurations around `fast-resnet`

### [Mixup](https://arxiv.org/abs/1710.09412)

We remove cutout data augmentation and uses mixup technics:
```
polyaxon run -u -f plx_configs/fastresnet/xp_training_mixup.yaml --name=xp_training_mixup --tags=original,mixup
```

### [AdamW](https://arxiv.org/pdf/1711.05101.pdf)

We uses decoupled weight decay Adam optimizer instead of SGD
```
polyaxon run -u -f plx_configs/fastresnet/xp_training_adamw.yaml --name=xp_training_adamw--tags=original,adamw
```

## Run hyperparameter tuning

```
polyaxon run -u -f plx_configs/fastresnet/gp_hp_bo_training.yaml --name=gp_hp_bo_training --tags=lt_20
``` 
or on WRN model
```
polyaxon run -u -f plx_configs/wrn/gp_hp_bo_training_wrn.yaml --name=gp_hp_bo_training_wrn --tags=lt_20,wrn
```



## Experiments on Google Colab

https://colab.research.google.com/drive/1W1_WEtatzyn32aPSrp4t5n66PuHQW6W8
