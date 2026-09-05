import numpy as np
def range_rmse(pred, gt): return np.sqrt(np.mean((np.array(pred)-np.array(gt))**2))
def az_wrap_loss(pred, gt):
    diff=(np.array(pred)-np.array(gt)+180)%360-180
    return np.sqrt(np.mean(diff**2))
def calibration_ece(conf, acc, n_bins=10):
    bins=np.linspace(0,1,n_bins+1)
    ece=0
    for i in range(n_bins):
        m=(conf>=bins[i])&(conf<bins[i+1])
        if m.sum()==0: continue
        ece+= abs(conf[m].mean()-acc[m].mean())*m.mean()
    return float(ece)
