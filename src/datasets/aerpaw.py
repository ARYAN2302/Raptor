"""AERPAW-28 stub — processed radar+truth only (§5). Real raw IQ not assumed."""
# AERPAW-28 provides radar position estimates + UAV ground truth (33 flights).
# We expose it as state labels for sim2real eval, not raw IQ supervision.
import json
def load_aerpaw_manifest(path): return json.loads(open(path).read()) if path else []
