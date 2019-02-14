import os
import argparse
from collections import OrderedDict
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

import ignite
from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import Timer
from ignite._utils import convert_tensor

from models import FastResnet, seq_conv_bn, conv_bn_elu    
from dataflow import get_fast_train_test_loaders
from dataflow import DynamicCrop, FlipLR, DynamicCutout
from custom_schedulers import get_layerwise_lr_scheduler

from polyaxon_client.tracking import Experiment, get_outputs_path


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def warmup_cudnn(model, criterion, batch_size):
    # run forward and backward pass of the model on a batch of random inputs
    # to allow benchmarking of cudnn kernels
    x = torch.Tensor(np.random.rand(batch_size, 3, 32, 32)).cuda().half()
    y = torch.LongTensor(np.random.randint(0, 10, batch_size)).cuda()

    model.train(True)
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    model.zero_grad()
    torch.cuda.synchronize()


def run(config, plx_experiment):

    set_seed(config['seed'])

    device = "cuda"
    batch_size = config['batch_size']
    
    train_loader, test_loader = get_fast_train_test_loaders(path=config["data_path"],
                                                            batch_size=batch_size,
                                                            num_workers=config['num_workers'],
                                                            device=device)

    bn_kwargs = config['bn_kwargs']
    conv_kwargs = config['conv_kwargs']

    model = FastResnet(conv_kwargs=conv_kwargs, bn_kwargs=bn_kwargs, final_weight=config['final_weight'])
    model = model.to(device).half()

    weight_decay = config['weight_decay']
    optimizer = optim.SGD(model.parameters(), lr=0.0,
                          momentum=config['momentum'],
                          weight_decay=weight_decay,
                          nesterov=True)

    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)    
    criterion = criterion.half()

    opt_kwargs = [
        ("lr", 0.0),
        ("momentum", config['momentum']),
        ("weight_decay", weight_decay),
        ("nesterov", True)
    ]
    optimizer = optim.SGD(
        [
            # conv + bn
            dict([("params", model.prep.parameters())] + opt_kwargs),
            # conv + bn
            dict([("params", model.layer1[0].parameters())] + opt_kwargs),
            # identity residual block
            dict([("params", model.layer1[-1].conv1.parameters())] + opt_kwargs),
            dict([("params", model.layer1[-1].conv2.parameters())] + opt_kwargs),
            # conv + bn
            dict([("params", model.layer2.parameters())] + opt_kwargs),
            # conv + bn
            dict([("params", model.layer3[0].parameters())] + opt_kwargs),
            # identity residual block
            dict([("params", model.layer3[-1].conv1.parameters())] + opt_kwargs),
            dict([("params", model.layer3[-1].conv2.parameters())] + opt_kwargs),
            # linear
            dict([("params", model.classifier.parameters())] + opt_kwargs),
        ]
    )

    num_iterations_per_epoch=len(train_loader)
    num_iterations = num_iterations_per_epoch * config['num_epochs']
    layerwise_lr_values = []
    layerwise_milestones = []
    for i in range(len(optimizer.param_groups)):
        key = "lr_param_group_{}".format(i)
        assert key in config, "{} not in config".format(key)
        data = config[key]
        layerwise_lr_values.append([data[0] / batch_size, 
                                    data[1] / batch_size, 
                                    data[3] / batch_size])
        layerwise_milestones.append([0.0, int(data[2]) * num_iterations_per_epoch, num_iterations])
    
    lr_scheduler = get_layerwise_lr_scheduler(optimizer, layerwise_lr_values, layerwise_milestones)
    def _prepare_batch_fp16(batch, device, non_blocking):
        x, y = batch
        return (convert_tensor(x, device=device, non_blocking=non_blocking).half(),
                convert_tensor(y, device=device, non_blocking=non_blocking).long())

    def process_function(engine, batch):
        x, y = _prepare_batch_fp16(batch, device=device, non_blocking=True)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()

        optimizer.step()
        loss = loss.item()

        return loss

    trainer = Engine(process_function)

    metrics = {
        "accuracy": Accuracy(),
        "loss": Loss(criterion) / len(test_loader)
    }
    evaluator = create_supervised_evaluator(model, metrics,
                                            prepare_batch=_prepare_batch_fp16,
                                            device=device, non_blocking=True)

    total_timer = Timer(average=False)
    train_timer = Timer(average=False)
    test_timer = Timer(average=False)

    if config["use_tb_logger"]:
        path = "experiments/tb_logs" if "TB_LOGGER_PATH" not in os.environ else os.environ["TB_LOGGER_PATH"]
        tb_logger = SummaryWriter(log_dir=path)

    test_timer.attach(evaluator, start=Events.EPOCH_STARTED)

    @trainer.on(Events.STARTED)
    def on_training_started(engine):
        print("Warming up cudnn on random inputs")
        for _ in range(5):
            for size in [batch_size, len(test_loader.dataset) % batch_size]:
                warmup_cudnn(model, criterion, size)

        total_timer.reset()

    @trainer.on(Events.EPOCH_STARTED)
    def on_epoch_started(engine):
        model.train()
        train_timer.reset()
        train_loader.dataset.set_random_choices()

    if config["use_tb_logger"]:
        @trainer.on(Events.ITERATION_COMPLETED)
        def on_iteration_completed(engine):
            # log learning rate
            param_name = "lr"
            if len(optimizer.param_groups) == 1:
                param = float(optimizer.param_groups[0][param_name])
                tb_logger.add_scalar(param_name,
                                     param * batch_size,
                                     engine.state.iteration)
            else:
                for i, param_group in enumerate(optimizer.param_groups):
                    param = float(param_group[param_name])
                    tb_logger.add_scalar("{}_group_{}".format(param_name, i),
                                         param * batch_size,
                                         engine.state.iteration)

            # log training loss
            tb_logger.add_scalar("training/loss_vs_iterations",
                                 engine.state.output / batch_size,
                                 engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def on_epoch_completed(engine):
        trainer.state.train_time = train_timer.value()

        if config["use_tb_logger"]:
            # Log |w|^2 and gradients
            for i, p in enumerate(model.parameters()):
                tb_logger.add_scalar("w2_{}_{}".format(i, list(p.data.shape)),
                                     torch.norm(p.data),
                                     engine.state.epoch)
                tb_logger.add_scalar("mean_grad_{}_{}".format(i, list(p.grad.shape)), 
                                     torch.mean(p.grad),
                                     engine.state.epoch)

        for i, p in enumerate(model.parameters()):
            plx_experiment.log_metrics(step=engine.state.epoch, 
                **{"w2_{}_{}".format(i, list(p.data.shape)):
                    torch.norm(p.data).item()})

        evaluator.run(test_loader)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, lr_scheduler)

    @evaluator.on(Events.COMPLETED)
    def log_results(engine):
        evaluator.state.test_time = test_timer.value()
        metrics = evaluator.state.metrics
        output = [(key, trainer.state.param_history[key][-1][0] * batch_size)
                    for key in trainer.state.param_history]
        output += [
            ("train time", trainer.state.train_time),
            ("train loss", trainer.state.output / batch_size),
            ("test time", evaluator.state.test_time),
            ("test loss", metrics['loss'] / batch_size),
            ("test acc", metrics['accuracy']),
            ("total time", total_timer.value())
        ]
        output = OrderedDict(output)        
        plx_experiment.log_metrics(step=trainer.state.epoch, **output)

        if config["use_tb_logger"]:
            tb_logger.add_scalar("training/loss",
                                 trainer.state.output / batch_size,
                                 trainer.state.epoch)
            tb_logger.add_scalar("training/total_time",
                                 total_timer.value(),
                                 trainer.state.epoch)
            tb_logger.add_scalar("test/loss",
                                 metrics['loss'] / batch_size,
                                 trainer.state.epoch)
            tb_logger.add_scalar("test/accuracy",
                                 metrics['accuracy'],
                                 trainer.state.epoch)

    trainer.run(train_loader, max_epochs=config['num_epochs'])

    if config["use_tb_logger"]:
        tb_logger.close()

if __name__ == "__main__":    

    parser = argparse.ArgumentParser("CIFAR10-fast")
    parser.add_argument('--use_tb_logger', action="store_true", default=False,
                        help="Use TensorBoard logger")
    parser.add_argument('--params', nargs="+", type=str,
                        help='Override default configuration with parameters: '
                        'batch_size=64 num_workers=12 ...')

    args = parser.parse_args()

    plx_experiment = Experiment()

    print("Run CIFAR10-fast")
    print("- PyTorch version: {}".format(torch.__version__))
    print("- Ignite version: {}".format(ignite.__version__))

    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    print("CUDA version: {}".format(torch.version.cuda))

    plx_experiment.log_params(**{
        "pytorch version": torch.__version__,
        "ignite version": ignite.__version__,
        "cuda version": torch.version.cuda
    })

    batch_size = 512
    num_epochs = 12
    config = {
        
        "data_path": ".",
        "seed": 12,
        
        "final_weight": 0.125,
        "conv_bn_fn": "seq_conv_bn",
        "bn_kwargs": {},
        "conv_kwargs": {},

        "momentum": 0.9,
        "weight_decay": 5e-4 * batch_size,

        "batch_size": batch_size,
        "num_workers": 2,

        "num_epochs": num_epochs,
        "use_tb_logger": args.use_tb_logger,

        "lr_param_group_0": [
            np.random.uniform(0.0, 1.0),  # start lr
            np.random.uniform(0.1, 5.0),  # peak lr
            np.random.randint(0, num_epochs),   # peak lr milestone
            np.random.uniform(0.0, 1.0),  # end lr
        ],
        "lr_param_group_1": [
            np.random.uniform(0.0, 1.0),  # start lr
            np.random.uniform(0.1, 5.0),  # peak lr
            np.random.randint(0, num_epochs),   # peak lr milestone
            np.random.uniform(0.0, 1.0),  # end lr
        ],
        "lr_param_group_2": [
            np.random.uniform(0.0, 1.0),  # start lr
            np.random.uniform(0.1, 5.0),  # peak lr
            np.random.randint(0, num_epochs),   # peak lr milestone
            np.random.uniform(0.0, 1.0),  # end lr
        ],
        "lr_param_group_3": [
            np.random.uniform(0.0, 1.0),  # start lr
            np.random.uniform(0.1, 5.0),  # peak lr
            np.random.randint(0, num_epochs),   # peak lr milestone
            np.random.uniform(0.0, 1.0),  # end lr
        ],
        "lr_param_group_4": [
            np.random.uniform(0.0, 1.0),  # start lr
            np.random.uniform(0.1, 5.0),  # peak lr
            np.random.randint(0, num_epochs),   # peak lr milestone
            np.random.uniform(0.0, 1.0),  # end lr
        ],
        "lr_param_group_5": [
            np.random.uniform(0.0, 1.0),  # start lr
            np.random.uniform(0.1, 5.0),  # peak lr
            np.random.randint(0, num_epochs),   # peak lr milestone
            np.random.uniform(0.0, 1.0),  # end lr
        ],
        "lr_param_group_6": [
            np.random.uniform(0.0, 1.0),  # start lr
            np.random.uniform(0.1, 5.0),  # peak lr
            np.random.randint(0, num_epochs),   # peak lr milestone
            np.random.uniform(0.0, 1.0),  # end lr
        ],
        "lr_param_group_7": [
            np.random.uniform(0.0, 1.0),  # start lr
            np.random.uniform(0.1, 5.0),  # peak lr
            np.random.randint(0, num_epochs),   # peak lr milestone
            np.random.uniform(0.0, 1.0),  # end lr
        ],
        "lr_param_group_8": [
            np.random.uniform(0.0, 1.0),  # start lr
            np.random.uniform(0.1, 5.0),  # peak lr
            np.random.randint(0, num_epochs),   # peak lr milestone
            np.random.uniform(0.0, 1.0),  # end lr
        ]
    }

    # Override config:
    if args.params:
        for param in args.params:
            key, value = param.split("=")
            if "/" not in value:
                value = eval(value)
            config[key] = value

    print("\n")
    print("Configuration:")
    for key, value in config.items():
        print("\t{}: {}".format(key, value))
    print("\n")

    plx_experiment.log_params(**config)

    run(config, plx_experiment)
