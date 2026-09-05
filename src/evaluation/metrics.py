import numpy as np
def range_rmse(p,g): return float(np.sqrt(np.mean((np.array(p)-np.array(g))**2)))
def az_rmse(p,g):
    d=(np.array(p)-np.array(g)+180)%360-180
    return float(np.sqrt(np.mean(d**2)))
