# -*- coding: utf-8 -*-
"""
@author: William
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 14

alpha = 2

name = [r"$p_{1}$",r"$p_{2}$",r"$p_{3}$",r"$p_{4}$",r"$p_{5}$"]
y = []

for i in range(10):
    v = np.random.default_rng().beta(1, alpha, 5)
    vv = (1 - v[: -1]).cumprod()
    p = np.empty_like(v)
    p[0] = v[0]
    p[1:-1] = v[1:-1] * vv[:-1]
    p[-1] = vv[-1]
    
    y.append(list(p))
    

t = range(10, 0, -1)
t = list(map(lambda x: str(x), t))

df = pd.DataFrame(data = y, columns = name, index = t)


ax = df.plot.barh(stacked=True, edgecolor='none')
plt.legend(df.columns)

horiz_offset = 1.01
vert_offset = 1.
ax.legend(bbox_to_anchor=(horiz_offset, vert_offset))
