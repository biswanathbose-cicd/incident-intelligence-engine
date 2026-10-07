"""CLI: ``python -m incident_engine.cli "vpn drops every hour" --train 5000``"""
from __future__ import annotations

import argparse

from .data import generate_incidents
from .pipeline import TriagePipeline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Triage an incident ticket (demo model trained on synthetic data)")
    ap.add_argument("text")
    ap.add_argument("--train", type=int, default=5000)
    ap.add_argument("-k", type=int, default=3)
    a = ap.parse_args(argv)
    r = TriagePipeline().fit(generate_incidents(a.train)).triage(a.text, a.k)
    print(f"category: {r.category} ({r.category_confidence:.2f})  priority: {r.priority} ({r.priority_confidence:.2f})")
    print("because:", ", ".join(t for t, _ in r.top_terms))
    for s in r.similar:
        print(f"  {s.similarity:.2f} {s.incident_id} [{s.category}] {s.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
