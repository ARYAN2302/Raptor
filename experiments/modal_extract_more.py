import modal
app = modal.App("raptor-extract-more")
image = modal.Image.debian_slim().run_commands(
    "apt-get update && apt-get install -y wget gnupg",
    "echo 'deb http://deb.debian.org/debian bookworm non-free non-free-firmware' >> /etc/apt/sources.list && apt-get update && apt-get install -y unrar p7zip-rar"
).add_local_dir("/tmp/raptor_build/Raptor", remote_path="/root/raptor")
data_vol = modal.Volume.from_name("raptor-data", create_if_missing=False)
@app.function(image=image, volumes={"/data": data_vol}, timeout=3600, cpu=2)
def extract():
    import subprocess, pathlib
    for rar in ["FLYSKY FS I6X.rar", "FRSKY X9DP2019.rar"]:
        p=f"/data/rfuav_rar/{rar}"
        out="/data/rfuav"
        print(f"extracting {rar} size {pathlib.Path(p).stat().st_size/1e9:.2f} GB")
        res=subprocess.run(["unrar", "x", "-o+", p, out+"/"], capture_output=True, text=True)
        print(res.stdout[:2000])
        print(res.stderr[:2000])
        print(f"done {rar}")
    for p in pathlib.Path("/data/rfuav").rglob("*.xml"):
        if "FLYSKY" in str(p) or "FRSKY" in str(p):
            print(p, open(p).read()[:400])
            break
    for p in list(pathlib.Path("/data/rfuav").rglob("*.iq"))[:5]:
        print(p, p.stat().st_size/1e6)
    from modal import Volume
    Volume.from_name("raptor-data").commit()
    print("committed 2 more drones")

