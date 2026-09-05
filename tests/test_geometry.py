import numpy as np
from src.simulation.geometry import array_steering
def test_steering():
    pos=np.zeros((4,3)); pos[:,0]=np.arange(4)*0.06
    w=array_steering(pos, 90, 10, 2.4e9)
    assert w.shape==(4,)
