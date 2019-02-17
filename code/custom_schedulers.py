from ignite.contrib.handlers import create_lr_scheduler_with_warmup, \
    LinearCyclicalScheduler, ConcatScheduler, CosineAnnealingScheduler, LRScheduler


def get_lr_scheduler(optimizer, num_iterations_per_epoch, config):
    lr_max_value = config['lr_max_value']
    warmup_duration = config['warmup_duration'] * num_iterations_per_epoch
    num_iterations = config['num_epochs'] * num_iterations_per_epoch
    cooldown_duration = config['cooldown_duration'] * num_iterations_per_epoch

    scheduler_1 = LinearCyclicalScheduler(optimizer, "lr",
                                          start_value=lr_max_value, end_value=lr_max_value * 0.4,
                                          cycle_size=(num_iterations - warmup_duration - cooldown_duration) * 2)

    scheduler_2 = LinearCyclicalScheduler(optimizer, "lr",
                                          start_value=lr_max_value * 0.2, end_value=lr_max_value * 0.01,
                                          cycle_size=cooldown_duration * 2)

    lr_scheduler = ConcatScheduler(schedulers=[scheduler_1, scheduler_2, ],
                                   durations=[num_iterations - warmup_duration - cooldown_duration, ])

    return create_lr_scheduler_with_warmup(
        lr_scheduler,
        warmup_start_value=0.0,
        warmup_end_value=lr_max_value,
        warmup_duration=warmup_duration,
        save_history=True,
    )


def get_piecewise_linear_lr_scheduler(optimizer, num_iterations_per_epoch, config):
    batch_size = config['batch_size']
    values = [v / batch_size for v in config['values']]
    milestones = [m * num_iterations_per_epoch for m in config['milestones']]
    return PiecewiseLinear(optimizer, "lr",
                           values=values,
                           milestones=milestones,
                           save_history=True)


def get_momentum_scheduler(optimizer, num_iterations_per_epoch, config):

    warmup_duration = config['warmup_duration'] * num_iterations_per_epoch
    num_iterations = config['num_epochs'] * num_iterations_per_epoch
    cooldown_duration = config['cooldown_duration'] * num_iterations_per_epoch

    scheduler_1 = LinearCyclicalScheduler(optimizer, "momentum",
                                          start_value=0.0, end_value=0.9,
                                          cycle_size=warmup_duration * 2)

    scheduler_2 = LinearCyclicalScheduler(optimizer, "momentum",
                                          start_value=0.9, end_value=0.9,
                                          cycle_size=(num_iterations - warmup_duration - cooldown_duration) * 2)


    scheduler_3 = LinearCyclicalScheduler(optimizer, "momentum",
                                          start_value=0.9, end_value=0.5,
                                          cycle_size=cooldown_duration * 2)

    momentum_scheduler = ConcatScheduler(schedulers=[scheduler_1, scheduler_2, scheduler_3],
                                         durations=[warmup_duration,
                                                    num_iterations - warmup_duration - cooldown_duration, ])

    return momentum_scheduler


def get_warmup_multistep_scheduler(optimizer, num_iterations_per_epoch, config):

    lr_max_value = config['lr_max_value']
    warmup_duration = config['warmup_duration'] * num_iterations_per_epoch
    num_iterations = config['num_epochs'] * num_iterations_per_epoch
    cooldown_duration = config['cooldown_duration'] * num_iterations_per_epoch

    optimizer.param_groups[0]['initial_lr'] = lr_max_value

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                   milestones=[
                                                        int(num_iterations - warmup_duration - 2 * cooldown_duration),
                                                        int(num_iterations - warmup_duration - 1 * cooldown_duration),
                                                        int(num_iterations - warmup_duration - 0.5 * cooldown_duration),
                                                   ],
                                                   gamma=0.2)

    return create_lr_scheduler_with_warmup(
        lr_scheduler,
        warmup_start_value=0.0,
        warmup_end_value=lr_max_value,
        warmup_duration=warmup_duration,
        save_history=True,
    )


def get_groupwise_lr_scheduler(optimizer, num_iterations_per_epoch, config):
    
    num_iterations = config['num_epochs'] * num_iterations_per_epoch
    batch_size = config['batch_size']

    def _get_lr_scheduler(o, lr, num_iterations):
        # values = [0.0, lr, 0.5 * lr, 0.35 * lr, 0.01 * lr]
        # milestones = [0, 4 * num_iterations_per_epoch, 10 * num_iterations_per_epoch, 10 * num_iterations_per_epoch, num_iterations]
        
        # values = [0.1 * lr, lr, 0.0 * lr, 0.05 * lr]
        # milestones = [0, 4 * num_iterations_per_epoch, num_iterations - 300, num_iterations]

        values = [0.1 * lr, lr, 0.0 * lr]
        milestones = [0, 4 * num_iterations_per_epoch, num_iterations]

        # values = [0.0, lr, 0.01 * lr, 0.3 * lr, 0.0]
        # milestones = [0, 4 * num_iterations_per_epoch, 9 * num_iterations_per_epoch, 9 * num_iterations_per_epoch, num_iterations]

        return PiecewiseLinear(o, param_name="lr", values=values, milestones=milestones, save_history=True)

    # create 5 schedulers for layers: prep, layer1, layer2, layer3 and classifier
    assert len(optimizer.param_groups) == 5
    lr_schedulers = []
    names = ['prep', 'layer1', 'layer2', 'layer3', 'classifier']
    # momentum = 0.9
    lrs = [2.15, 0.3, 0.4, 0.7, 0.7]
    # lrs = [2.15, 0.3, 0.4, 1.0, 0.7]

    # # momentum = 0.3
    # lrs = [3.0, 4.0, 5.0, 9.0, 7.0]

    # # momentum = 0.5
    # lrs = [1.0, 2.0, 3.0, 7.0, 5.0]

    for i, param_group in enumerate(optimizer.param_groups):
        lr_schedulers.append(_get_lr_scheduler(param_group,
                                               lrs[i] / batch_size,
                                               num_iterations))

    return ParamGroupScheduler(
        schedulers=lr_schedulers,
        names=names
    )


def get_layerwise_lr_scheduler(optimizer, milestones_lr_values):

    def _get_lr_scheduler(o, milestones_lr_values):
        return PiecewiseLinear(o, param_name="lr", milestones_values=milestones_lr_values, save_history=True)

    lr_schedulers = []    
    names = []
    for i, param_group in enumerate(optimizer.param_groups):
        names.append("lr_group_{}".format(i))
        lr_schedulers.append(_get_lr_scheduler(param_group, milestones_lr_values[i]))

    return ParamGroupScheduler(
        schedulers=lr_schedulers,
        names=names
    )


def get_layerwise_scheduler(optimizer, param_name, milestones_values):

    def _get_scheduler(o, pn, mv):
        return PiecewiseLinear(o, param_name=pn, milestones_values=mv, save_history=True)

    schedulers = []
    names = []
    for i, param_group in enumerate(optimizer.param_groups):
        names.append("{}_group_{}".format(param_name, i))
        schedulers.append(_get_scheduler(param_group, param_name, milestones_values[i]))

    return ParamGroupScheduler(
        schedulers=schedulers,
        names=names
    )


# #############################################################################
# PiecewiseLinear parameter scheduler
# #############################################################################

from ignite.contrib.handlers.param_scheduler import ParamScheduler

try:
    from collections.abc import Sequence
except ImportError:  # Python 2.7 compatibility
    from collections import Sequence
    from itertools import izip as zip


class ParamGroupScheduler(object):
    """
    Scheduler helper to group multiple schedulers into one. 

    Args:
        schedulers (list of ParamScheduler): list of parameter schedulers.
        names (list of str): list of names of schedulers.
    
    .. code-block:: python

        names = []
        lr_schedulers = []
        for i, param_group in enumerate(optimizer.param_groups):
            names.append("param_group_{}".format(i))
            lr_schedulers.append(get_lr_scheduler(param_group, param_group_scheduling_conf[i]))

        scheduler = ParamGroupScheduler(schedulers=lr_schedulers, names=names)
        # Attach single scheduler to the trainer
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    """
    def __init__(self, schedulers, names):
        if len(names) != len(schedulers):
            raise RuntimeError("{} should be equal {}".format(len(schedulers), len(names)))
        self.schedulers = schedulers
        self.names = names

    def __call__(self, engine):
        for scheduler, name in zip(self.schedulers, self.names):
            scheduler(engine, name=name)


class PiecewiseLinear(ParamScheduler):
    """
    Piecewise linear parameter scheduler

    Args:
        optimizer (`torch.optim.Optimizer` or dict): the optimizer or parameters group to use.
        param_name (str): name of optimizer's parameter to update.
        milestones_values (list of tuples (int, float)): list of tuples (event index, parameter value)
            represents milestones and parameter. Milestones should be increasing integers.
        save_history (bool, optional): whether to log the parameter values to
            `engine.state.param_history`, (default=False).

    Returns:
        PiecewiseLinear: piecewise linear scheduler


    .. code-block:: python

        scheduler = PiecewiseLinear(optimizer, "lr",
                                    milestones_values=[(10, 0.5), (20, 0.45), (21, 0.3), (30, 0.1), (40, 0.1)])
        # Attach to the trainer
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
        #
        # Sets the learning rate to 0.5 over the first 10 iterations, then decreases linearly from 0.5 to 0.45 between
        # 10th and 20th iterations. Next there is a jump to 0.3 at the 21st iteration and LR decreases linearly
        # from 0.3 to 0.1 between 21st and 30th iterations and remains 0.1 until the end of the iterations.
        #
    """

    def __init__(self, optimizer, param_name, milestones_values, save_history=False):
        super(PiecewiseLinear, self).__init__(optimizer, param_name, save_history)

        if not isinstance(milestones_values, Sequence) or len(milestones_values) < 1:
            raise ValueError("Argument milestones_values should be a list or tuple with at least one value, "
                             "but given {}".format(type(milestones_values)))

        values = []
        milestones = []
        for pair in milestones_values:
            if not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError("Argument milestones_values should be a list of pairs (milestone, param_value)")
            if not isinstance(pair[0], int):
                raise ValueError("Value of a milestone should be integer, but given {}".format(type(pair[0])))
            if len(milestones) > 0 and pair[0] < milestones[-1]:
                raise ValueError("Milestones should be increasing integers, but given {} is smaller "
                                 "than the previous milestone {}".format(pair[0], milestones[-1]))
            milestones.append(pair[0])
            values.append(pair[1])

        self.values = values
        self.milestones = milestones
        self._index = 0

    def _get_start_end(self):
        if self.milestones[0] > self.event_index:
            return self.event_index - 1, self.event_index, self.values[0], self.values[0]
        elif self.milestones[-1] <= self.event_index:
            return self.event_index, self.event_index + 1, self.values[-1], self.values[-1],
        elif self.milestones[self._index] <= self.event_index < self.milestones[self._index + 1]:
            return self.milestones[self._index], self.milestones[self._index + 1], \
                self.values[self._index], self.values[self._index + 1]
        else:
            self._index += 1
            return self._get_start_end()

    def get_param(self):
        start_index, end_index, start_value, end_value = self._get_start_end()
        return start_value + (end_value - start_value) * (self.event_index - start_index) / (end_index - start_index)
