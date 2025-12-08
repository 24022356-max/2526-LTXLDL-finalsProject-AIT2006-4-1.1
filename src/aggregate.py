
import pandas as pd
import numpy as np
import glob
import os
import gc

Input = "../raw"
Output = "../proccesed"
Year = 2019





clean_files = sorted(glob.glob(os.path.join(Output, 'clean_yellow_tripdata_2019-*.parquet')))

