import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys


# Set BASE_DIR to the absolute path of the file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define STATE_SPACES_DIR and add it to sys.path
STATE_SPACES_DIR = os.path.join(BASE_DIR, "state-spaces")
sys.path.insert(0, STATE_SPACES_DIR)

# Define STATE_SPACES_MODELS_DIR and add it to sys.path
STATE_SPACES_MODELS_DIR = os.path.join(STATE_SPACES_DIR, "models")
sys.path.insert(0, STATE_SPACES_MODELS_DIR)


# Import S4D from the s4 package
from s4.s4d import S4D

class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        lr=0.001,
        dropout_fn=nn.Dropout1d,

    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)

        """

        x = x.unsqueeze(-1) # added by us to comply with shape requirement (not from s4's original src code)

        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x

    @staticmethod
    def setup_optimizer(model, lr, weight_decay, epochs):
        """
        S4 requires a specific optimizer setup.
        The S4 layer (A, B, C, dt) parameters typically
        require a smaller learning rate (typically 0.001), with no weight decay.
        The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
        and weight decay (if desired).
        """

        # All parameters in the model
        all_parameters = list(model.parameters())

        # General parameters don't contain the special _optim key
        params = [p for p in all_parameters if not hasattr(p, "_optim")]

        # Create an optimizer with the general parameters
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
        ]  # Unique dicts
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **hp}
            )

        # Create a lr scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        # Print optimizer info
        keys = sorted({k for hp in hps for k in hp.keys()})
        for i, g in enumerate(optimizer.param_groups):
            group_hps = {k: g.get(k, None) for k in keys}
            print(' | '.join([
                                 f"Optimizer group {i}",
                                 f"{len(g['params'])} tensors",
                             ] + [f"{k} {v}" for k, v in group_hps.items()]))

        return optimizer, scheduler
