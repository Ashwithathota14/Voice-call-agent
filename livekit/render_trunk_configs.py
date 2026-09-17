"""Fill outbound-trunk.json / inbound-trunk.json with real RingTrunk credentials
from pipecat/.env, without ever putting those credentials in a file git tracks.

outbound-trunk.json and inbound-trunk.json in this folder stay placeholders —
safe to commit. This script reads the real trunk id / password / source IP /
number from pipecat/.env (gitignored) and writes outbound-trunk.local.json /
inbound-trunk.local.json (also gitignored) next to them. Point `lk` at the
*.local.json files, not the placeholder ones.

Usage (from repo root or livekit/):
    python render_trunk_configs.py             # renders both
    python render_trunk_configs.py outbound    # outbound only — no RINGTRUNK_SOURCE_IP needed
    python render_trunk_configs.py inbound     # inbound only

    lk sip outbound create livekit/outbound-trunk.local.json
    lk sip inbound create livekit/inbound-trunk.local.json
    lk sip dispatch create livekit/dispatch-rule.json   # no secrets, use as-is

Requires in pipecat/.env: RINGTRUNK_TRUNK_ID, RINGTRUNK_SIP_PASSWORD,
RINGTRUNK_SIP_SERVER, CALLER_ID — plus RINGTRUNK_SOURCE_IP only for inbound.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE.parent / "pipecat" / ".env")


def require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is empty in pipecat/.env — fill it in first")
    return value


def main() -> None:
    import sys

    only = sys.argv[1] if len(sys.argv) > 1 else "both"  # "outbound" | "inbound" | "both"

    trunk_id = require("RINGTRUNK_TRUNK_ID")
    sip_password = require("RINGTRUNK_SIP_PASSWORD")
    sip_server = require("RINGTRUNK_SIP_SERVER")
    number = require("CALLER_ID")

    if only in ("outbound", "both"):
        outbound = {
            "trunk": {
                "name": "RingTrunk outbound",
                "address": sip_server,
                "transport": "SIP_TRANSPORT_TCP",
                "numbers": [number],
                "auth_username": trunk_id,
                "auth_password": sip_password,
            }
        }
        (HERE / "outbound-trunk.local.json").write_text(json.dumps(outbound, indent=2) + "\n")
        print("wrote livekit/outbound-trunk.local.json")
        print("next: lk sip outbound create livekit/outbound-trunk.local.json")

    if only in ("inbound", "both"):
        source_ip = require("RINGTRUNK_SOURCE_IP")  # only needed for inbound
        inbound = {
            "trunk": {
                "name": "RingTrunk inbound",
                "numbers": [number, f"+91{number}", f"0{number}", f"91{number}"],
                "allowed_addresses": [source_ip],
            }
        }
        (HERE / "inbound-trunk.local.json").write_text(json.dumps(inbound, indent=2) + "\n")
        print("wrote livekit/inbound-trunk.local.json")
        print("next: lk sip inbound create livekit/inbound-trunk.local.json")


if __name__ == "__main__":
    main()
