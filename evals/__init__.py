"""Evaluation harness for AI-Uptime (Phase 3, R4 role).

Mirrors the sibling repo's ``evals/`` convention: a corpus module (synthetic
facilities with adjudicated by-construction labels), a pure metrics module,
no-leakage time-split fold utilities, named fault-injection scenarios, and a
threshold-gated runner that ``make eval`` invokes. Sparse-label honesty
applies everywhere: every headline metric carries a confidence interval and
the report states explicitly that all labeled cases are synthetic.
"""
