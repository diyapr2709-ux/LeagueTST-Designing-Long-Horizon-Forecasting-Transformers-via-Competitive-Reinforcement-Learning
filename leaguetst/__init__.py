"""LeagueTST: long-horizon forecasting transformers designed by competition.

The package is organised by stage of the project:

* ``leaguetst.data``: benchmark loading, splitting, scaling and windowing.
* ``leaguetst.models``: the configurable PatchTST forecaster and DLinear.
* ``leaguetst.training``: the training loop and evaluation metrics.
* ``leaguetst.search``: the design space and, later, the search methods.
* ``leaguetst.utils``: seeding, device selection and file helpers.
"""

__version__ = "0.1.0"
