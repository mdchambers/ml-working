import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

df = pd.read_csv("~/diamonds.csv")

# print(df)
# plt.show()

x = np.arange(10).reshape((5,2))
y = pd.DataFrame(x)