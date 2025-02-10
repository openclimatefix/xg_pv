"""Basic types used in package"""

from typing import Union

import numpy as np
import pandas as pd

Hour = int
Features = pd.DataFrame
ProcessedNWP = Union[np.ndarray, pd.DataFrame]
