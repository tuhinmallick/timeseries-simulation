"""
This file is a copy from the same file name in commodity-desk-modeling repo. 
For the up-to-date STLModel please always check the file in the original repo. 
"""

from statsmodels.tsa.seasonal import STL
import os, sys, pathlib
import numpy as np
import pandas as pd
import logging

utils_location = pathlib.Path(__file__).absolute().parent
if os.path.realpath(utils_location) not in sys.path:
    sys.path.append(os.path.realpath(utils_location))

# turn off INFO message print from target logger
logger = logging.getLogger("fbprophet")
logger.setLevel(logging.WARN)


class STLModel:
    # will be used as model dictionary keys and data dictionary prefix
    mystl_name = pd.Index(["trend", "season", "arma"])

    def _make_STL(
        self,
        data: pd.DataFrame,
        target: list,
        freq=None,
        rolling_strategy="rolling",
        rolling=24,
        extract_strategy="nanmean",
        **kwargs
    ):
        """
        This function is called by process_data() to generate the 3 features of the target: trend, season, arma.
        :target: the target feature that will be decomposed by the STL method.
        :test_size: see process_data() args.
        :rolling_strategy: str - see process_data() the ``stl_strategy`` argument.
        :rolling: int - the rolling window over seasonal_decompose() to generate the 3 parts.
        :kwargs: dict - additional argument setting for STL method
        :extract_strategy: str - see process_data() the ``stl_extract_strategy`` argument.
        """
        #!!!: core algorithm here to play around 1. with the setting of STL(), and 2.
        # with the strategy of collecting decomposed values.

        # Insight: 1. both STL & seasonal_decompose are very sensitive to `period`.
        # 2: decomposed values in the boundary of rolling window is more error-prone since no data at the opposite side of the current evaluated point.
        # 3. refer to _mask_STL_insight.ipynb to see why I use nanmean method to collect the groun truth decomposed values (welcome to improve this part)

        # freq
        if freq is None:
            freq = pd.infer_freq(data.index)
        data.index.freq = freq
        # stl method
        final = []
        for tar in target:
            subli = []
            # case for initial window
            period = STL(data[tar]).config["period"]
            seasonal = period + (period % 2 == 0)  # Ensure being odd
            obj = STL(data[tar].iloc[:rolling], seasonal=seasonal, **kwargs).fit()
            obj = pd.DataFrame(
                dict(zip(self.mystl_name, [obj.trend, obj.seasonal, obj.resid]))
            )
            subli.append(obj)
            # case starting the rolling/expanding window
            for i in range(1, len(data) - rolling + 1):
                if rolling_strategy == "rolling":
                    start = i
                elif rolling_strategy == "expand":
                    start = 0
                tmp = STL(
                    data[tar].iloc[start : i + rolling], seasonal=seasonal, **kwargs
                ).fit()
                subli.append(
                    pd.DataFrame(
                        dict(zip(self.mystl_name, [tmp.trend, tmp.seasonal, tmp.resid]))
                    )
                )
            # compile final df for the current target
            collector = []
            for key in self.mystl_name:  # trend/season/arma
                tmpconcated = pd.concat([subdf[key] for subdf in subli], axis=1)
                if extract_strategy == "nanmean":
                    collector.append(np.nanmean(tmpconcated, axis=1))
                elif extract_strategy == "last":
                    collector.append(tmpconcated.bfill(axis=1).iloc[:, 0])
            collector = np.array(collector).T  # after .T => sample_size x 3
            collector = pd.DataFrame(
                collector, index=data.index, columns=tar + "_" + self.mystl_name
            )
            final.append(collector)
        # add back with original data columns
        output = pd.concat([data, *final], axis=1)
        return output
