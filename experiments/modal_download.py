import modal
app = modal.App("raptor-download")
image = modal.Image.debian_slim().apt_install("unrar").pip_install("requests","tqdm").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
data_vol = modal.Volume.from_name("raptor-data", create_if_missing=False)
@app.function(image=image, volumes={"/data": data_vol}, timeout=3600, cpu=4)
def extract_mini4():
    import subprocess, pathlib, os
    rar="/data/rfuav_rar/DJI MINI4 PRO.rar"
    out="/data/rfuav/DJI MINI4 PRO"
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    print(f"extracting {rar} -> {out} size {pathlib.Path(rar).stat().st_size/1e9:.2f} GB")
    # unrar e or x
    res=subprocess.run(["unrar", "x", "-o+", rar, out+"/"], capture_output=True, text=True)
    print(res.stdout[:2000])
    print(res.stderr[:2000])
    print("done, listing:")
    for p in pathlib.Path(out).rglob("*"):
        if p.is_file():
            print(f"{p} {p.stat().st_size/1e6:.1f} MB")
            if "xml" in p.suffix:
                print(open(p).read()[:500])
                break
    data_vol.commit()
    print("committed")
