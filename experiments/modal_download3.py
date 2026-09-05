import modal
app = modal.App("raptor-download3")
image = modal.Image.debian_slim().apt_install("p7zip-full").pip_install("requests").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
data_vol = modal.Volume.from_name("raptor-data", create_if_missing=False)
@app.function(image=image, volumes={"/data": data_vol}, timeout=3600, cpu=2)
def extract():
    import subprocess, pathlib
    rar="/data/rfuav_rar/DJI MINI4 PRO.rar"
    out="/data/rfuav"
    print(f"rar {rar} exists {pathlib.Path(rar).exists()}")
    if pathlib.Path(rar).exists():
        print(f"size {pathlib.Path(rar).stat().st_size/1e9:.2f} GB")
    # 7z
    res=subprocess.run(["7z", "x", rar, f"-o{out}", "-y"], capture_output=True, text=True)
    print(res.stdout[:5000])
    print(res.stderr[:2000])
    for p in pathlib.Path(out).rglob("*.xml"):
        print(p)
        if "MINI4" in str(p):
            print(open(p).read()[:800])
            break
    for p in list(pathlib.Path(out).rglob("*.iq"))[:5]:
        print(p, p.stat().st_size/1e6)
    try: 
        import modal
        vol=modal.Volume.from_name("raptor-data")
        vol.commit()
    except: pass
    print("done")
