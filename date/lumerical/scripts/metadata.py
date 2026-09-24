"""Protocol-v2 result provenance metadata.

Every protocol-v2 run writes a `metadata` block into its result.json /
fresh_sim.json so that headline tables can trace each value back to the exact
script, bugfix level, verifier, and scheduler that produced it.
"""
import hashlib

PROTOCOL_VERSION = "v2"

# Bugfix flags — reflect the three protocol bugs fixed before the v2 rerun.
BUGFIX_FLAGS = [
    "reject_not_committed",   # Bug 1: reject pixel never enters committed/pool
    "resume_ec_bi_restore",   # Bug 3: resume restores ec/bi from checkpoint
    "sign_threshold_zero",    # Bug 2: sign gate uses threshold=0, never -inf
]

RERUN_DIR = "final_rerun_2026_08_13_protocol_v2"


def script_hash(path):
    """Short sha1 of the script file, for provenance."""
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def make_metadata(script_path, verifier_type, scheduler_type, **extra):
    """Build the standard metadata block for a result.json / fresh_sim.json."""
    meta = {
        "protocol_version": PROTOCOL_VERSION,
        "script_hash": script_hash(script_path),
        "bugfix_flags": BUGFIX_FLAGS,
        "verifier_type": verifier_type,
        "scheduler_type": scheduler_type,
        "rerun_dir": RERUN_DIR,
    }
    meta.update(extra)
    return meta
