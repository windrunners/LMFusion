from typing import Tuple, Optional, Callable
import torch
from torch.optim.optimizer import Optimizer

# functions
def exists(val):
    return val is not None

# update functions
def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # step weight decay
    p.data.mul_(1 - lr * wd)          # p=p*(1-lr*0.01)  lr = 0.0001

    # weight update
    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1-beta1)  # update=[(beta1)*exp_avg+(1-beta1)*grad]

    update = torch.where(update == 0, torch.tensor(0.0),
                torch.where((update > 0) & (update < 0.001), torch.tensor(0.5),
                            torch.where(update >= 0.001, torch.tensor(1.0),
                                        torch.where((update > -0.001) & (update < 0), torch.tensor(-0.5),
                                                    torch.tensor(-1.0)))))  # update=piecewise function(update)

    p.add_(update, alpha=-lr)   # p=p-lr*update=p-lr*{sign[(beta1)*exp_avg+(1-beta1)*grad]}

    # decay the momentum running average coefficient
    exp_avg.mul_(beta2).add_(grad, alpha=1-beta2)    # exp_avg=(beta2)*exp_avg+(1-beta2)*grad

class piecewise(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_triton: bool = False
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])

        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay
        )

        super().__init__(params, defaults)

        self.update_fn = update_fn

        if use_triton:
            from lion_pytorch.triton import update_fn as triton_update_fn
            self.update_fn = triton_update_fn

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]

                # init state - exponential moving average of gradient values
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                self.update_fn(p, grad, exp_avg, lr, wd, beta1, beta2)

        return loss
