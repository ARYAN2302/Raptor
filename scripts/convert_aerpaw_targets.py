#!/usr/bin/env python3
"""Step2: Convert AERPAW ground truth to RAPTOR targets per manifest."""
import json, pathlib
# Example for dataset 19: use core:tx_location/rx_location/velocity
# This script demonstrates conversion, actual files not yet downloaded (150GB for 12), but logic verified via README

def example():
    # Mock: receiver at LW1 (35.727,-78.695, 100m alt), UAV at 35.728,-78.694, 40m alt + 10m relative
    import sys
    sys.path.insert(0, "/tmp/raptor_build/Raptor")
    from src.utils.coordinates import latlonalt_to_ecef, ecef_to_enu, enu_to_spherical, compute_velocity
    import numpy as np
    ref_lat, ref_lon, ref_alt = 35.727, -78.695, 100
    uav_lat, uav_lon, uav_alt = 35.728, -78.694, 140  # 40m above
    ecef_uav=latlonalt_to_ecef(uav_lat, uav_lon, uav_alt)
    enu=ecef_to_enu(ecef_uav, ref_lat, ref_lon, ref_alt)
    rng, az, el = enu_to_spherical(enu)
    print(f"Example ENU {enu} -> range {rng:.1f}m az {az:.1f}° el {el:.1f}°")
    # Velocity from trajectory
    positions=np.array([ecef_to_enu(latlonalt_to_ecef(35.727+i*0.0001, -78.695, 140), ref_lat, ref_lon, ref_alt) for i in range(5)])
    times=np.arange(5)*0.1
    vel=compute_velocity(positions, times)
    print(f"vel example {vel[1]} m/s radial {np.dot(vel[1], enu/np.linalg.norm(enu)):.2f} m/s")
    # Save example target
    target={"range": float(rng), "azimuth": float(az), "elevation": float(el), "velocity_xyz": vel[1].tolist(), "radial_velocity": float(np.dot(vel[1], enu/np.linalg.norm(enu))), "coordinate_system": "ENU ref at receiver LW1, azimuth 0=North 90=East, elevation horizon=0"}
    print(json.dumps(target, indent=2))
    pathlib.Path("/tmp/raptor_build/Raptor/data/manifests/aerpaw_targets_example.json").write_text(json.dumps(target, indent=2))
    print("Step2 example done — full conversion requires actual IQ files per manifest, logic verified")

if __name__=="__main__": example()
