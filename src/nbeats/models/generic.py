from .base import NBeatsModelBase
from ..stacks import GenericStack

class NBeatsGeneric(NBeatsModelBase):
    def __init__(
        self,
        n_stacks,
        n_blocks,
        backcast,
        forecast,
        n_layers=4,
        n_theta=4,
        hidden_dim=None,
        shared_weights=True
    ):
        self.n_stacks = n_stacks
        self.n_blocks = n_blocks
        self.backcast = backcast
        self.forecast = forecast
        self.n_layers = n_layers
        self.n_theta = n_theta
        self.hidden_dim = hidden_dim
        self.shared_weights = shared_weights
        super().__init__()

    def _build_stacks(self):
        return [
            GenericStack(
                n_blocks=self.n_blocks,
                backcast=self.backcast,
                forecast=self.forecast,
                n_layers=self.n_layers,
                n_theta=self.n_theta,
                hidden_dim=self.hidden_dim,
                shared_weights=self.shared_weights
            )
            for _ in range(self.n_stacks)
        ]

