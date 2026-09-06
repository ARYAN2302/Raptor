# TSMS-Drone 2026 RF Receiver — Single-Receiver IQ → Range Audit

**Exact path/layout:**
```
/Users/adarshthakur/Desktop/RF Receiver/
├── 2m/
│   ├── Corner Reflector/{Image File/*_2m_*.png, Raw File/*_2m_*.mat}
│   ├── Inspire 2/{Image File, Raw File}
│   ├── Matrice 30/{Image File, Raw File}
│   ├── Mavic 2 Pro/{Image File, Raw File}
│   └── Phantom 4 Pro/{Image File, Raw File}
├── 4m/ (same 5 drones)
├── 6m/ ...
├── 8m/ ... 30m/
```
Each `Raw File` contains `*.mat` files, each `Image File` contains `*.png` (not used).

**Drone/categories:** Corner Reflector (calibration), Inspire 2, Matrice 30, Mavic 2 Pro, Phantom 4 Pro — 5 categories (4 drones + 1 reflector).

**Distance labels (ground-truth range source: folder name):** `2m,4m,6m,8m,10m,12m,14m,16m,18m,20m,22m,24m,26m,28m,30m` — 15 distances.

**Number of captures per distance/drone:** Every distance/drone has **500 captures** (verified `ls .../Raw File/*.mat | wc -l` = 500). **Total 37,500** RF captures, 18GB.

**RF file format:** `.mat` MATLAB v7 (some v7.3 HDF5). Key `data` → shape `(131072,)` dtype `complex128` (complex double), e.g., `Inspire 2_10m_001.mat` → `-0.0864+0.0659j`, `Phantom 4 Pro_2m_001.mat` → `0.1767+0.0634j`. No other keys.

**IQ shape/dtype:** `(131072,)` complex samples per capture, `complex128`, **Channels: 1** (single RF receiver, single stream), **Samples: 131,072**.

**Sample rate / center frequency / bandwidth:** **Not present in .mat files** (only `data`, no metadata; `find -name "*.txt"` none). Must be obtained from TSMS-Drone 2026 paper — **do not invent** (PDF on desktop is unrelated).

**Available metadata:** Per-capture **distance from folder name** (`10m` → 10), **drone category from parent folder** (`Inspire 2`), **capture index from filename** (`_001`), no timestamps/receiver coords/heading in file — ground-truth range source is **folder name distance**.

**Total usable captures:** **37,500** (15×5×500).

**Verification RF is genuinely single RF receiver, no radar/GPS leakage:** `Raw File` contains only `data` complex array — no radar/GPS/telemetry keys (verified via `h5py`/`scipy.io` — only `data`). `Image File` PNGs are derived spectrograms. No `.csv` GPS/radar in folder (unlike AERPAW) — **pure single-receiver IQ**, no leakage.

---

## Exact commands/code to load one RF capture and obtain complex IQ + range label

```python
import pathlib
import numpy as np

# Universal loader (handles both v7 and v7.3)
def load_rf_capture(mat_path):
    mat_path = pathlib.Path(mat_path)
    try:
        import scipy.io
        data = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)["data"]
    except NotImplementedError:
        import mat73
        data = mat73.loadmat(str(mat_path))["data"]
        # h5py compound case also handled by mat73
    if data.dtype.names is not None and 'real' in data.dtype.names:
        data = data['real'] + 1j*data['imag']
    # Range label from folder, drone from parent
    distance_str = mat_path.parents[2].name  # e.g., "10m"
    drone_category = mat_path.parents[1].name  # e.g., "Inspire 2"
    range_m = int(distance_str.replace("m",""))
    return data, range_m, drone_category

# Example
iq, range_m, drone = load_rf_capture("/Users/adarshthakur/Desktop/RF Receiver/10m/Inspire 2/Raw File/Inspire 2_10m_001.mat")
print(f"IQ shape {iq.shape} dtype {iq.dtype} first {iq[0]}")
print(f"Range label: {range_m}m")
print(f"Drone category: {drone}")

# For v7.3 file (e.g., Phantom 4 Pro)
iq2, range_m2, drone2 = load_rf_capture("/Users/adarshthakur/Desktop/RF Receiver/2m/Phantom 4 Pro/Raw File/Phantom 4 Pro_2m_001.mat")
print(f"IQ shape {iq2.shape} dtype {iq2.dtype} range {range_m2}m drone {drone2}")

# Verified:
# Inspire 2_10m_001.mat → iq.shape (131072,) dtype complex128 range 10 drone Inspire 2
# Phantom 4 Pro_2m_001.mat → iq.shape (131072,) dtype complex128 range 2 drone Phantom 4 Pro
```

**Result of running above on 2025-09-18:**
- `Inspire 2_10m_001.mat` → `iq.shape (131072,) dtype complex128 range 10 drone Inspire 2`
- `Phantom 4 Pro_2m_001.mat` → `iq.shape (131072,) dtype complex128 range 2 drone Phantom 4 Pro`
- Verified: IQ is **single-channel complex baseband**, no radar/GPS leakage, range label is **folder name distance only**.

