import os.path

from ..models.graph.model import ContextAwareArtRecSys
from .train_graph import train
from .EarlyStopping import EarlyStopping
import torch
from hyperopt import hp, Trials, fmin, STATUS_OK, tpe, space_eval


class FineTuner:
    def __init__(self, data, train_data, val_data, device=torch.device('cuda') if torch.cuda.is_available() else 'cpu',
                 space=None,
                 out_dir='./'):
        self.data = data
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.trials = None
        self.space = space if space else FineTuner.get_space()
        self.root=out_dir
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    @staticmethod
    def get_space():
        return {
            "gnn_layers": hp.choice("gnn_layers", [1, 2, 3]),
            "hidden_channels": hp.choice("hidden_channels", [128, 256, 512, 768]),
            "drop_rate": hp.choice("drop_rate", [0.2, 0.4, 0.5, 0.6])
        }

    def hyperparameter_tuning(self, params):
        model = ContextAwareArtRecSys(self.train_data.metadata(), hidden_channels=params["hidden_channels"],
                                      gnn_layers=params["gnn_layers"],
                                      drop_rate=params['drop_rate'],
                                      activation=torch.nn.ReLU()).to(self.device)

        name_model = 'gnn_model_hidden_channels-{hidden_channels}_gnn_layers-{gnn_layers}_drop_rate-{drop_rate}.pt'
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2, verbose=False)
        early_stop = EarlyStopping(patience=7, min_delta=1e-5, checkpoint_path=self.root+name_model.format(**params))
        metrics_to_watch = {}

        val_loss = train(self.train_data.to(self.device), self.val_data.to(self.device),
                         early_stop, scheduler, 50, model, criterion, optimizer, metrics_to_watch, verbose=False,
                         device=self.device)
        return {"loss": val_loss, "status": STATUS_OK}

    def tune(self, max_evals=100):
        self.trials = Trials()
        space = self.space
        fn = self.hyperparameter_tuning
        best = fmin(
            fn=fn,
            space=space,
            algo=tpe.suggest,
            trials=self.trials,
            max_evals=max_evals
        )
        return space_eval(space, best)
