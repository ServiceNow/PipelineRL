import torch
from peft.peft_model import PeftModel
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from transformers import Adafactor, PreTrainedModel


def get_grouped_params(
    model: PreTrainedModel | PeftModel,
    weight_decay: float,
    no_decay: list[str] = ["bias", "LayerNorm.weight"],
):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        # Skip frozen parameters
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def get_optimizer(name, model, learning_rate, weight_decay):
    grouped_params = get_grouped_params(model, weight_decay)
    match name:
        case "adamw_torch":
            optimizer = AdamW(grouped_params, lr=learning_rate)
        case "adafactor":
            optimizer = Adafactor(
                grouped_params,
                lr=learning_rate,
                relative_step=False,
                scale_parameter=False,
            )
        case "cpuadam":
            import deepspeed.ops.adam

            optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(grouped_params, lr=learning_rate)
        case "lion":
            optimizer = Lion(grouped_params, lr=learning_rate)
        case _:
            raise ValueError(f"Unknown optimizer: {name}")
    return optimizer


class Lion(Optimizer):
    r"""PyTorch implementation of the Lion optimizer from https://github.com/google/automl/blob/master/lion/lion_pytorch.py"""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialize the hyperparameters.

        Args:
          params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
          lr (float, optional): learning rate (default: 1e-4)
          betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
          weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.

        Returns:
          (tensor): the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
