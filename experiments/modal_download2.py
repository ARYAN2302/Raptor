import modal
app = modal.App("raptor-download2")
image = modal.Image.debian_slim().apt_install("unrar-free").pip_install("requests","tqdm").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
data_vol = modal.Volume.from_name("raptor-data", create_if_missing=False)
@app.function(image=image, volumes={"/data": data_vol}, timeout=3600, cpu=4)
def extract_mini4():
    import subprocess, pathlib, os
    rar="/data/rfuav_rar/DJI MINI4 PRO.rar"
    out="/data/rfuav/DJI MINI4 PRO"
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    print(f"rar exists {pathlib.Path(rar).exists()} size {pathlib.Path(rar).stat().st_size/1e9:.2f} GB" if pathlib.Path(rar).exists() else "no rar")
    # try unrar-free
    for cmd in [["unrar-free", "x", rar, out+"/"], ["unar", rar, "-o", out]]:
        print(f"try {cmd}")
        res=subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout[:2000]); print(res.stderr[:2000])
        if res.returncode==0: break
    for p in list(pathlib.Path(out).rglob("*"))[:20]:
        print(p, p.stat().st_size if p.is_file() else "dir")
    data_vol.commit()
