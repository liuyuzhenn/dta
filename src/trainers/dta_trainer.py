import time

import numpy as np
import torch
import yaml
from tqdm import tqdm

from ..losses.base_loss import NoGradientError
from ..losses.hornsmethod import horns_method
from ..losses.utils import compute_nmse_np
from .base_trainer import BaseTrainer
from .utils import *


class DtaTrainer(BaseTrainer):
    def __init__(self, model, dataset, loss):
        super().__init__(model, dataset, loss)

    def test(self, test_configs):
        """
        Test model after training.

        Args:
            test_configs: arguments for testing

        Returns:
            None
        """
        self.device = test_configs["device"]
        checkpoint = torch.load(
            test_configs["checkpoint"], map_location=test_configs["device"]
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        val_data = self.dataset.get_data_loader(test_configs["split"])

        self.model.eval()
        # Validation
        avg_meter = DictAverageMeter()
        with torch.no_grad():
            self.model.eval()
            for data in tqdm(val_data):
                data = to_device(data, self.device)
                model_outputs = self.model.forward(data)

                try:
                    loss = self.loss_term.compute(model_outputs, data)
                except NoGradientError:
                    continue

                if isinstance(loss, tuple):
                    loss, items = loss
                else:
                    items = None

                if items is not None:
                    items.update({"loss": float(loss)})
                else:
                    items = {"loss": float(loss)}

                metrics = self._metrics(model_outputs, data)
                if metrics is not None:
                    items.update(metrics)

                avg_meter.update(tensor2float(items))
        metrics = avg_meter.mean()
        print(metrics)
        with open(test_configs["file_path"], "w") as f:
            yaml.dump(metrics, f, default_flow_style=False)

    def _metrics(self, outputs_model, inputs_data, mode="train") -> dict:
        with torch.no_grad():
            t_pred = outputs_model["t"].cpu().numpy().astype(np.float64)
            t_gt = inputs_data["t_gt"][0].cpu().numpy().astype(np.float64)

            nrmse = compute_nmse_np(t_pred, t_gt) ** 0.5
            dist = np.sum((t_pred - t_gt) ** 2, axis=-1) ** 0.5

            metrics = {
                "nrmse": float(nrmse),
                "mean": float(np.mean(dist)),
                "median": float(np.median(dist)),
            }

        return metrics
