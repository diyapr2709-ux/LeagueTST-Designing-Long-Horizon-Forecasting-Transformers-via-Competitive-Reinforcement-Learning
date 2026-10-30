"""Design space and search methods.

Only the design space is implemented at this stage. Random search,
regularized evolution, the architect league and the referee come later.
"""

from leaguetst.search.space import SEARCH_SPACE, enumerate_configs, sample_config, space_size

__all__ = ["SEARCH_SPACE", "enumerate_configs", "sample_config", "space_size"]
