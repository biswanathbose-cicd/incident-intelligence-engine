"""Seeded synthetic ITSM incident generator.

No real ticket data is used anywhere in this repository. Every incident belongs to a *problem*
(a shared root cause: same service, symptom and error code), which gives retrieval a ground truth:
two incidents are "similar" iff they share ``problem_id``. Noise is injected on purpose (typos,
shared filler text, cross-category mentions, noisy severity cues) so accuracy is not trivially 100%."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

# category -> (error-code prefix, services, symptoms)
CATEGORIES: dict[str, tuple[str, list[str], list[str]]] = {
    "network": ("NET", ["vpn gateway", "wifi controller", "dns resolver", "load balancer", "firewall",
                        "proxy server", "wan link"],
                ["connection timed out", "packet loss", "intermittent drops", "high latency",
                 "unable to reach", "dns lookup failing", "tunnel keeps disconnecting"]),
    "email": ("MAIL", ["exchange server", "mail relay", "outlook client", "calendar sync",
                       "distribution list", "mailbox quota"],
              ["emails stuck in outbox", "bounce back errors", "cannot send attachments",
               "delayed delivery", "sync not working", "mailbox full"]),
    "identity_access": ("IAM", ["sso portal", "active directory", "mfa service", "password reset portal",
                                "badge system", "ldap directory"],
                        ["account locked out", "password expired", "mfa prompt not arriving",
                         "access denied", "cannot login", "permission missing"]),
    "hardware": ("HW", ["laptop", "docking station", "monitor", "printer", "keyboard", "mobile phone",
                        "conference room display"],
                 ["will not power on", "screen flickering", "battery draining fast", "paper jam",
                  "overheating", "not detected"]),
    "software": ("SW", ["crm application", "erp client", "office suite", "browser extension",
                        "video conferencing app", "code editor", "reporting tool"],
                 ["keeps crashing", "fails to launch", "upgrade failed", "license expired",
                  "freezes on save", "blank screen"]),
    "database": ("DB", ["orders database", "reporting warehouse", "postgres cluster", "oracle instance",
                        "data pipeline job", "replication"],
                 ["slow queries", "connection pool exhausted", "deadlock detected", "replication lag",
                  "disk space low", "backup failed"]),
    "security": ("SEC", ["endpoint protection", "phishing filter", "vulnerability scanner", "siem alert",
                         "certificate store", "data loss prevention"],
                 ["suspicious login detected", "malware flagged", "certificate expired",
                  "unusual outbound traffic", "phishing email reported", "policy violation"]),
}
PREFIXES = [v[0] for v in CATEGORIES.values()]
FILLER = ["hi team", "please help", "this started this morning", "urgent", "thanks", "not sure what changed",
          "happening for several users", "started after the update", "need assistance", "reported by a colleague",
          "still ongoing", "was working yesterday"]
SEVERITY_CUES = {
    "P1": ["complete outage", "all users affected", "production down", "business critical"],
    "P2": ["multiple users impacted", "major degradation", "key team blocked"],
    "P3": ["single user affected", "workaround available", "intermittent"],
    "P4": ["minor issue", "cosmetic", "general question", "low impact"],
}
PRIORITIES = ["P1", "P2", "P3", "P4"]


@dataclass(frozen=True)
class Incident:
    id: str
    text: str
    category: str
    priority: str
    problem_id: int


@dataclass(frozen=True)
class _Problem:
    category: str
    service: str
    symptom: str
    code: str
    severity: str
    host: str


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 5:
        return word
    i = rng.randint(1, len(word) - 2)
    return word[:i] + word[i + 1:] if rng.random() < 0.5 else word[:i] + word[i + 1] + word[i] + word[i + 2:]


def _make_problems(n: int, rng: random.Random) -> list[_Problem]:
    cats = list(CATEGORIES)
    out = []
    for _ in range(n):
        cat = rng.choice(cats)
        prefix, services, symptoms = CATEGORIES[cat]
        code_prefix = prefix if rng.random() < 0.5 else rng.choice(PREFIXES)  # legacy codes mislead sometimes
        severity = rng.choices(PRIORITIES, weights=[5, 20, 45, 30])[0]
        out.append(_Problem(cat, rng.choice(services), rng.choice(symptoms),
                            f"{code_prefix}-{rng.randint(1000, 9999)}", severity, f"srv{rng.randint(1, 60):02d}"))
    return out


def _render(p: _Problem, rng: random.Random) -> tuple[str, str]:
    clauses: list[str] = []
    r = rng.random()
    if r < 0.07:
        pass  # vague ticket: no service or symptom at all, only code / filler / severity cues
    elif r < 0.67:
        clauses.append(f"{p.service} {p.symptom}")
    elif r < 0.82:
        clauses.append(p.service)
    else:
        clauses.append(p.symptom)
    if rng.random() < 0.7:
        clauses.append(f"error {p.code}")
    if rng.random() < 0.5:
        clauses.append(f"host {p.host}")
    # severity cue, sometimes taken from a neighbouring level (label noise on purpose)
    level = PRIORITIES.index(p.severity)
    if rng.random() < 0.15:
        level = min(3, max(0, level + rng.choice([-1, 1])))
    if rng.random() < 0.65:
        clauses.append(rng.choice(SEVERITY_CUES[PRIORITIES[level]]))
    if rng.random() < 0.3:  # cross-category confuser
        other = rng.choice([c for c in CATEGORIES if c != p.category])
        clauses.append(f"also {rng.choice(CATEGORIES[other][2])}")
    clauses.extend(rng.sample(FILLER, rng.choice([0, 1, 1, 2])))
    if not clauses:
        clauses.append(rng.choice(FILLER))  # a ticket always says *something*
    rng.shuffle(clauses)
    words = " ".join(clauses).split()
    words = [_typo(w, rng) if rng.random() < 0.04 else w for w in words]
    priority = p.severity
    if rng.random() < 0.2:  # the recorded priority is itself noisy
        priority = PRIORITIES[min(3, max(0, PRIORITIES.index(priority) + rng.choice([-1, 1])))]
    return " ".join(words), priority


def generate_incidents(n: int = 20_000, seed: int = 7, incidents_per_problem: int = 25) -> list[Incident]:
    rng = random.Random(seed)
    n_problems = max(2, n // incidents_per_problem)
    problems = _make_problems(n_problems, rng)
    weights = 1.0 / (np.arange(n_problems) + 1.0) ** 0.6  # a few big recurring problems, a long tail
    pids = np.random.default_rng(seed).choice(n_problems, size=n, p=weights / weights.sum())
    out = []
    for i, pid in enumerate(pids):
        text, priority = _render(problems[pid], rng)
        out.append(Incident(f"INC{i:07d}", text, problems[pid].category, priority, int(pid)))
    return out
