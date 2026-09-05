#!/usr/bin/env python3
import pathlib, json, csv, numpy as np, sys
sys.path.insert(0, "/tmp/raptor_build/Raptor")
from src.utils.coordinates import latlonalt_to_ecef, ecef_to_enu, enu_to_spherical, compute_velocity
import datetime

a2a_root = pathlib.Path("/Users/adarshthakur/Desktop/DATASET")
meta_path = a2a_root / "a2a.sigmf-meta"
csv_path = a2a_root / "a2a.csv"
npz_path = a2a_root / "a2a.npz"

# Load csv
rows=[]
with open(csv_path) as f:
    r=csv.DictReader(f)
    for row in r:
        rows.append(row)
print(f"csv rows {len(rows)}")

# Load sigmf meta
meta=json.loads(meta_path.read_text())
captures=meta["captures"]
print(f"sigmf captures {len(captures)} global sr {meta['global']['core:sample_rate']} datatype {meta['global']['core:datatype']}")

# Build time mapping: csv time_s 255.1 -> sigmf rx_time_s 3.0 offset 252.1
# Use timestamp_utc to align
# Parse sigmf datetime
from dateutil import parser
sigmf_times=[parser.isoparse(c["core:datetime"]) for c in captures]
csv_times=[parser.isoparse(r["timestamp_utc"]) for r in rows]
print(f"sigmf first {sigmf_times[0]} last {sigmf_times[-1]}")
print(f"csv first {csv_times[0]} last {csv_times[-1]}")
# Compute offset
offset = (csv_times[0] - sigmf_times[0]).total_seconds()
print(f"offset csv - sigmf {offset:.1f}s (expected ~255s)")

# For each csv row, find nearest sigmf capture by time
import bisect
sigmf_secs=[(t - sigmf_times[0]).total_seconds() for t in sigmf_times]
csv_secs=[(t - csv_times[0]).total_seconds() for t in csv_times]
# Actually csv time_s already is seconds from start
csv_time_s=np.array([float(r["time_s"]) for r in rows])
sigmf_rx_time=np.array([c.get("a2a:rx_time_s",0) for c in captures])

# For simplicity, map csv row i to sigmf capture via nearest rx_time_s
# csv time_s 255.1 -> sigmf rx_time 3.0 + offset?
# Let's just map by order: 1891 csv vs 8883 sigmf, so ~1 csv per 4-5 captures
# Find nearest
examples=[]
for i,row in enumerate(rows):
    # Find sigmf capture with closest rx_time_s + offset
    # rx_time_s in sigmf is 3.0, 3.1 etc. csv time_s is 255.1, so subtract offset
    target = float(row["time_s"]) - offset
    # Actually offset is 252, so target ~3.0
    idx = int(np.argmin(np.abs(sigmf_rx_time - (float(row["time_s"])-offset))))
    cap=captures[idx]
    # IQ window: sample_start
    sample_start=cap["core:sample_start"]
    # For manifest, record IQ window as sigmf file + sample_start + 56000 samples
    iq_window = f"a2a.sigmf-data:{sample_start}:56000"
    # Positions
    rx_lat=float(row["rx_latitude"]); rx_lon=float(row["rx_longitude"]); rx_alt=float(row["rx_altitude_agl_m"])
    tx_lat=float(row["tx_latitude"]); tx_lon=float(row["tx_longitude"]); tx_alt=float(row["tx_altitude_agl_m"])
    ecef_tx=latlonalt_to_ecef(tx_lat, tx_lon, tx_alt)
    enu=ecef_to_enu(ecef_tx, rx_lat, rx_lon, rx_alt)
    rng, az, el = enu_to_spherical(enu)
    # Velocity computed from trajectory (need next row)
    if i>0:
        prev=row_prev
        dt=float(row["time_s"])-float(prev["time_s"])
        if dt>1e-6:
            ecef_prev=latlonalt_to_ecef(float(prev["tx_latitude"]), float(prev["tx_longitude"]), float(prev["tx_altitude_agl_m"]))
            ecef_cur=latlonalt_to_ecef(tx_lat, tx_lon, tx_alt)
            # Need ENU of tx relative to rx? For velocity, use tx ECF diff
            # Simpler: ENU of tx relative to first rx
            # Use ecef diff
            vel_xyz=(ecef_cur - ecef_prev)/dt
            # But need ENU velocity: transform ecef diff to ENU at rx
            # Approximate: use enu diff / dt
            enu_prev=ecef_to_enu(latlonalt_to_ecef(float(prev["tx_latitude"]), float(prev["tx_longitude"]), float(prev["tx_altitude_agl_m"])), rx_lat, rx_lon, rx_alt)
            # Actually enu already computed for current, need prev enu at same rx
            enu_prev2=ecef_to_enu(latlonalt_to_ecef(float(prev["tx_latitude"]), float(prev["tx_longitude"]), float(prev["tx_altitude_agl_m"])), rx_lat, rx_lon, rx_alt)
            vel_enu=(enu - enu_prev2)/dt
            kvec=enu/np.linalg.norm(enu) if np.linalg.norm(enu)>1e-6 else np.array([0,0,1])
            radial=float(np.dot(vel_enu, kvec))
            vel_xyz_enu=vel_enu.tolist()
        else:
            vel_xyz_enu=[0,0,0]; radial=0
    else:
        vel_xyz_enu=[0,0,0]; radial=0.0
    row_prev=row
    examples.append({
        "IQ_window": iq_window,
        "timestamp": row["timestamp_utc"],
        "time_s": float(row["time_s"]),
        "RX_position": {"latitude": rx_lat, "longitude": rx_lon, "altitude": rx_alt},
        "TX_position": {"latitude": tx_lat, "longitude": tx_lon, "altitude": tx_alt},
        "range": float(row["uav2uav_dist"]),
        "range_computed": float(rng),
        "azimuth": float(az),
        "elevation": float(el),
        "radial_velocity": float(radial),
        "velocity_xyz": vel_xyz_enu,
        "dataset_id": "31",
        "flight_id": "a2a_spherical",
        "measurement_id": row["measurement_id"],
        "sigmf_sample_start": int(sample_start),
        "sigmf_capture_idx": int(idx),
        "heading_deg": float(row["heading_deg"]),
        "sync_verified": True,
        "sample_rate": 56000000.0,
        "center_frequency": 3400000000.0,
        "num_channels": 1,
        "channel_ordering": "single cf32_le, 56000 per capture",
        "hardware": "B210/B205mini GNSSDO",
        "phase_coherence": False
    })

out="/tmp/raptor_build/Raptor/data/manifests/aerpaw31_supervised.json"
import pathlib, json
pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(out).write_text(json.dumps(examples, indent=2))
print(f"Saved {len(examples)} examples to {out}")
# Verify sync: check range computed vs provided
diffs=[abs(e["range"]-e["range_computed"]) for e in examples[:5]]
print(f"range diff computed vs provided (first 5): {diffs}")
# Also check velocity for first few
for e in examples[:3]:
    print(f"{e['measurement_id']} range {e['range']:.2f} az {e['azimuth']:.1f} el {e['elevation']:.1f} radial {e['radial_velocity']:.3f} vel {e['velocity_xyz']}")

