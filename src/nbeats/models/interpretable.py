from .base import NBeatsModelBase
from ..stacks import TrendStack, SeasonalityStack

class NBeatsInterpretable(NBeatsModelBase):
    def __init__(
        self,
        n_blocks,
        backcast,
        forecast,
        n_layers=4,
        degree=4,
        n_harmonics=2,
        hidden_dim=None,
        shared_weights=True
    ):
        self.n_blocks = n_blocks
        self.backcast = backcast
        self.forecast = forecast
        self.n_layers = n_layers
        self.degree = degree
        self.n_harmonics = n_harmonics
        self.hidden_dim = hidden_dim
        self.shared_weights = shared_weights
        super().__init__()

    def _build_stacks(self):
        return [
            TrendStack(
                n_blocks=self.n_blocks,
                backcast=self.backcast,
                forecast=self.forecast,
                degree=self.degree,
                n_layers=self.n_layers,
                hidden_dim=self.hidden_dim,
                shared_weights=self.shared_weights
            ),
            SeasonalityStack(
                n_blocks=self.n_blocks,
                backcast=self.backcast,
                forecast=self.forecast,
                n_harmonics=self.n_harmonics,
                n_layers=self.n_layers,
                hidden_dim=self.hidden_dim,
                shared_weights=self.shared_weights
            )
        ]
