import json, hashlib, pathlib
def hash_manifest(path): return hashlib.sha256(open(path,"rb").read()).hexdigest()[:12]
def load_manifest(path): return json.loads(open(path).read())
