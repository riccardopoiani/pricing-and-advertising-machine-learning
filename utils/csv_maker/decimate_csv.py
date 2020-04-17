import os
import sys

import pandas as pd

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.append("../../")


COUNT_COL = "user"
CSV_FILE = "../../report/csv/pricing_bandit/Apr17_10-32-23/discrete_user_regret.csv"
TAKE_A_POINT_EVERY = 1000

df: pd.DataFrame = pd.read_csv(CSV_FILE)
df = df.iloc[::TAKE_A_POINT_EVERY, :]

print("New csv lenght: {}".format(df.shape))

df.to_csv("../../report/csv/pricing_bandit/Apr17_10-32-23/decimated_discrete_user_regret.csv".format(""), index=False)
