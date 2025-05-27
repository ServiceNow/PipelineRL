from typing import Callable

import torch
import transformers
from torch.utils.data.dataloader import DataLoader

from .context import get_accelerator, logger
from .types import TrainingMetrics


def evaluate(
    args,
    model: transformers.PreTrainedModel,
    eval_dataloader: DataLoader,
):
    model.eval()
    losses = []
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            loss = outputs.loss.repeat(args.valid_batch_size)
            losses.append(get_accelerator().gather(loss).detach())  # type: ignore
            if args.max_eval_steps > 0 and step >= args.max_eval_steps:
                break
    model.train()
    return torch.mean(torch.cat(losses)).item()


def evaluate_and_get_metrics(
    args, model, eval_dataloader, dev_dataloader, training_metrics: TrainingMetrics
):
    if eval_dataloader:
        logger.info("Evaluating model")
        training_metrics.eval_loss = evaluate(args, model, eval_dataloader)
        if dev_dataloader:
            training_metrics.dev_loss = evaluate(args, model, dev_dataloader)
        else:
            training_metrics.dev_loss = 0
        if training_metrics.eval_loss < training_metrics.best_eval_loss:
            training_metrics.best_eval_loss = training_metrics.eval_loss
            training_metrics.best_completed_steps = training_metrics.completed_steps
    return training_metrics


def dummy_eval_callback(config_name: str) -> Callable:
    def dummy(*args, **kwargs):
        return {}

    return dummy
