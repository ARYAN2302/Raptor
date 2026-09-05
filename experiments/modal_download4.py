import modal
app = modal.App("raptor-download4")
image = modal.Image.debian_slim().run_commands(
    "apt-get update && apt-get install -y wget gnupg",
    "echo 'deb http://deb.debian.org/debian bookworm non-free non-free-firmware' >> /etc/apt/sources.list && apt-get update && apt-get install -y unrar p7zip-rar || echo fail",
).pip_install("requests").add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
data_vol = modal.Volume.from_name("raptor-data", create_if_missing=False)
@app.function(image=image, volumes={"/data": data_vol}, timeout=3600, cpu=2)
def extract():
    import subprocess, pathlib
    rar="/data/rfuav_rar/DJI MINI4 PRO.rar"
    out="/data/rfuav"
    print(f"rar exists {pathlib.Path(rar).exists()} size {pathlib.Path(rar).stat().st_size/1e9:.2f} GB")
    for cmd in [["unrar", "x", "-o+", rar, out+"/"], ["7z", "x", rar, f"-o{out}", "-y"]]:
        print(f"try {cmd[0]}")
        res=subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout[:3000]); print(res.stderr[:2000])
        if res.returncode==0: break
    for p in pathlib.Path(out).rglob("DJI MINI4 PRO*"):
        print(p, p.stat().st_size/1e6 if p.is_file() else "dir")
        if p.suffix==".xml" and p.is_file():
            print(open(p).read()[:600])
            break
    for p in list(pathlib.Path(out).rglob("*.iq"))[:5]:
        print(p, p.stat().st_size/1e6)
    try:
        from modal import Volume
        Volume.from_name("raptor-data").commit()
    except: pass
