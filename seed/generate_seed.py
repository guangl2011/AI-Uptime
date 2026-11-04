"""Generate the Phase 0 seed: a trivial cold-chain example facility.

Phase 0 ships a hand-written stub inventory (JSON) so `make demo` has a
deterministic fixture. Phase 1 replaces this with the synthetic-corpus
generator from the agent-team plan (R4): cold-chain first, AIDC second.

Usage: python seed/generate_seed.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# A deliberately trivial facility: one cold room, its refrigeration rack,
# and the power feed — enough to exercise loading, nothing more.
STUB_FACILITY: dict[str, object] = {
    "facility_id": "cold_chain_example_01",
    "name": "Example Cold Storage — Phase 0 stub",
    "segment": "cold_chain",
    "assets": [
        {"asset_id": "COMP-01", "asset_type": "compressor", "location": "machine room"},
        {"asset_id": "COMP-02", "asset_type": "compressor", "location": "machine room"},
        {"asset_id": "RACK-01", "asset_type": "refrigeration_rack", "location": "machine room"},
        {"asset_id": "COLDROOM-1", "asset_type": "cold_room", "location": "freezer wing"},
        {"asset_id": "FEED-UPS-1", "asset_type": "ups_feed", "location": "electrical room"},
    ],
    "note": "Phase 0 stub — no graph, redundancy, or evidence yet.",
}


def write_seed(out_path: Path) -> Path:
    """Write the stub facility JSON deterministically."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(STUB_FACILITY, indent=2, sort_keys=True) + "\n"
    out_path.write_text(payload, encoding="utf-8")
    return out_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("seed/example_facility.json"))
    args = parser.parse_args()
    written = write_seed(args.out)
    print(f"[seed] wrote {written} ({len(STUB_FACILITY['assets'])} stub assets)")


if __name__ == "__main__":
    main()
