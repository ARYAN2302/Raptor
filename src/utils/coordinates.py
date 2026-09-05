"""Coordinate conversions for Step2 — explicit ENU frame."""
import numpy as np
import math
# WGS84
def latlonalt_to_ecef(lat_deg, lon_deg, alt_m):
    a=6378137.0; f=1/298.257223563; e2=2*f - f*f
    lat=np.deg2rad(lat_deg); lon=np.deg2rad(lon_deg)
    N=a/np.sqrt(1-e2*np.sin(lat)**2)
    x=(N+alt_m)*np.cos(lat)*np.cos(lon)
    y=(N+alt_m)*np.cos(lat)*np.sin(lon)
    z=(N*(1-e2)+alt_m)*np.sin(lat)
    return np.array([x,y,z])
def ecef_to_enu(ecef, ref_lat, ref_lon, ref_alt):
    # ref ecef
    ref_ecef=latlonalt_to_ecef(ref_lat, ref_lon, ref_alt)
    d=ecef-ref_ecef
    lat=np.deg2rad(ref_lat); lon=np.deg2rad(ref_lon)
    R=np.array([[-np.sin(lon), np.cos(lon), 0],
                [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                [ np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon), np.sin(lat)]])
    return R.dot(d)
def enu_to_spherical(enu):
    e,n,u=enu
    rng=np.linalg.norm(enu)
    az=np.rad2deg(np.arctan2(e,n))%360
    el=np.rad2deg(np.arctan2(u, np.hypot(e,n)))
    return rng, az, el
def compute_velocity(positions, timestamps):
    # positions [N,3] ENU, timestamps [N] seconds
    vel=np.zeros_like(positions)
    dt=np.diff(timestamps)
    # avoid div by zero
    for i in range(1,len(positions)):
        if dt[i-1]>1e-6:
            vel[i]=(positions[i]-positions[i-1])/dt[i-1]
    return vel
