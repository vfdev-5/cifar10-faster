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

## Produce original result

```
polyaxon run -u -f plx_configs/xp_original_training.yaml --name=xp_original_training --tags=original
```


