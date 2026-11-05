"""Sim M1 — the simulator core (art_non0bSFL, milestone M1).

A seeded, deterministic, discrete-event layer over the existing eval corpus
and scenario conventions. The assessor is a pure single-fault what-if engine
over an immutable graph snapshot — it has no clock and no state. The
simulator adds the missing layer: events mutate the working topology, the
real assessment engine re-runs on every resulting snapshot, and the full
assessment state is recorded as a trace step.

Modules (import from the concrete module, mirroring the repo's ``evals/``
convention of docstring-only package inits):

- ``engine``  — pure state transitions: events, cascade closure, materialization
- ``dsl``     — sequenced-scenario JSON parsing/validation (``evals/sim/scenarios/``)
- ``runner``  — drives the real API, derives observed trace steps, evaluates gates
- ``report``  — markdown/JSON report rendering (workflow artifacts)
- ``__main__``— ``python -m evals.sim`` CLI (``make sim``)

Governance (binding): the simulator drives the assessment engine only — no
ingestion connector, no external system, no write outside the repo's own
eval/sim output. What-if markers are an M3 extension (the DSL validates the
event-kind union and rejects unknown kinds with a named extension point);
they are deliberately not implemented here.
"""
