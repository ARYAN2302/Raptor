#!/usr/bin/env python3
"""Task3: Convert SigMF + telemetry to synchronized training examples per §9."""
import json, pathlib
import numpy as np
from src.utils.coordinates import latlonalt_to_ecef, ecef_to_enu, enu_to_spherical, compute_velocity

def process_a2g(a2g_root="/Users/adarshthakur/Desktop/A2G_Channel_Measurements", out="/tmp/raptor_build/Raptor/data/manifests/aerpaw19_train.json"):
    import pathlib, json, numpy as np
    root=pathlib.Path(a2g_root)
    flights=[p for p in root.iterdir() if p.is_dir()]
    examples=[]
    for flight in flights[:2]:  # small subset for audit
        metas=sorted(flight.glob("*.sigmf-meta"))[:5]
        for m in metas:
            j=json.loads(m.read_text())
            cap=j["captures"][0] if j.get("captures") else {}
            # IQ window: would load sigmf-data, but for manifest just record file
            iq_file=str(m).replace(".sigmf-meta",".sigmf-data")
            # Ground truth
            tx=cap.get("core:tx_location",{})
            rx=cap.get("core:rx_location",{})
            vel=cap.get("core:velocity",{})
            # Compute ENU
            if tx and rx:
                import sys
                sys.path.insert(0, "/tmp/raptor_build/Raptor")
                from src.utils.coordinates import latlonalt_to_ecef, ecef_to_enu, enu_to_spherical
                ecef_tx=latlonalt_to_ecef(tx["latitude"], tx["longitude"], tx["altitude"])
                enu=ecef_to_enu(ecef_tx, rx["latitude"], rx["longitude"], rx["altitude"])
                rng, az, el = enu_to_spherical(enu)
                # radial
                vel_vec=np.array([vel.get("velocity_x",0), vel.get("velocity_y",0), vel.get("velocity_z",0)])
                kvec=enu/np.linalg.norm(enu) if np.linalg.norm(enu)>1e-6 else np.array([0,0,1])
                radial=float(np.dot(vel_vec, kvec))
            else:
                rng=az=el=radial=None
                vel_vec=[0,0,0]
            examples.append({
                "IQ_window": iq_file,
                "timestamp": cap.get("core:timestamp"),
                "receiver_position": rx,
                "transmitter_position": tx,
                "range": float(rng) if rng else None,
                "azimuth": float(az) if az else None,
                "elevation": float(el) if el else None,
                "radial_velocity": radial,
                "velocity_xyz": [vel.get("velocity_x"), vel.get("velocity_y"), vel.get("velocity_z")],
                "dataset_id": "19",
                "flight_id": flight.name,
                "valid_masks": {"range": rng is not None, "azimuth": False, "elevation": False, "velocity": True}  # az/el not trainable per single channel
            })
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(examples, indent=2))
    print(f"A2G manifest {len(examples)} examples -> {out}")
    return examples

def process_a2a(a2a_root="/Users/adarshthakur/Desktop/DATASET", out="/tmp/raptor_build/Raptor/data/manifests/aerpaw31_train.json"):
    import pathlib, json, numpy as np, csv
    csv_path=pathlib.Path(a2a_root)/"a2a.csv"
    # Load a few rows
    examples=[]
    with open(csv_path) as f:
        import csv
        r=csv.DictReader(f)
        rows=list(r)[:5]
        for row in rows:
            # IQ window: a2a.sigmf-data offset via sample_start, but for manifest just record measurement_id
            iq_window=f"a2a.sigmf-data:{row['measurement_id']}"
            # Positions
            rx_lat=float(row["rx_latitude"]); rx_lon=float(row["rx_longitude"]); rx_alt=float(row["rx_altitude_agl_m"])
            tx_lat=float(row["tx_latitude"]); tx_lon=float(row["tx_longitude"]); tx_alt=float(row["tx_altitude_agl_m"])
            # Compute range/az/el via ENU
            import sys
            sys.path.insert(0, "/tmp/raptor_build/Raptor")
            from src.utils.coordinates import latlonalt_to_ecef, ecef_to_enu, enu_to_spherical
            ecef_tx=latlonalt_to_ecef(tx_lat, tx_lon, tx_alt)
            enu=ecef_to_enu(ecef_tx, rx_lat, rx_lon, rx_alt)
            rng, az, el = enu_to_spherical(enu)
            # Velocity not in csv, will be computed via diff later, for now None
            examples.append({
                "IQ_window": iq_window,
                "timestamp": row["timestamp_utc"],
                "receiver_position": {"latitude": rx_lat, "longitude": rx_lon, "altitude": rx_alt},
                "transmitter_position": {"latitude": tx_lat, "longitude": tx_lon, "altitude": tx_alt},
                "range": float(row["uav2uav_dist"]),
                "azimuth": float(az),
                "elevation": float(el),
                "radial_velocity": None,  # need computed
                "velocity_xyz": None,
                "dataset_id": "31",
                "flight_id": "a2a_spherical",
                "valid_masks": {"range": True, "azimuth": False, "elevation": False, "velocity": False}  # velocity not directly, az/el not trainable single channel
            })
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(examples, indent=2))
    print(f"A2A manifest {len(examples)} examples -> {out}")
    return examples

if __name__=="__main__":
    process_a2g()
    process_a2a()
    print("Task3 manifests ready — splits must be by flight/trajectory, e.g., train flights 2023-12-15_15_41,15_51 -> test 15_58, never random windows")
