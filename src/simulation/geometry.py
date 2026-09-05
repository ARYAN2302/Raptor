import numpy as np
def array_steering(pos, az_deg, el_deg, freq):
    c=3e8; lam=c/freq
    az=np.deg2rad(az_deg); el=np.deg2rad(el_deg)
    kvec=np.array([np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)])
    delay=pos.dot(kvec)/c
    return np.exp(1j*2*np.pi*freq*delay)
