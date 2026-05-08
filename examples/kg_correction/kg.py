"""Synthetic KG fixture — small, hand-curated, Wikidata-flavored.

A clean KG used as the "ground truth" before perturbation. Schema:

    {
      "entities": {
        "<entity_id>": {"label": "...", "type": "Person" | "Film"},
        ...
      },
      "edges": [
        {"id": "e<n>", "subject": "...", "predicate": "...", "object": "..."},
        ...
      ]
    }

Predicates and the (subject_type, object_type) tuples they require are
declared in ``ALLOWED_PREDICATES``. The structural critic uses this to
flag type violations.

This fixture is intentionally tiny so PoC runs are reproducible and
fast. Phase B will swap it for a Wikidata SPARQL fetch.
"""
from __future__ import annotations

import copy
from typing import Any


ALLOWED_PREDICATES: dict[str, tuple[str, str]] = {
    "directed":   ("Person", "Film"),
    "starred_in": ("Person", "Film"),
    "wrote":      ("Person", "Film"),
    "produced":   ("Person", "Film"),
    "spouse_of":  ("Person", "Person"),
}


CLEAN_KG: dict[str, Any] = {
    "entities": {
        "Q1":  {"label": "Bong Joon-ho",     "type": "Person"},
        "Q2":  {"label": "Park Chan-wook",   "type": "Person"},
        "Q3":  {"label": "Lee Chang-dong",   "type": "Person"},
        "Q4":  {"label": "Kim Jee-woon",     "type": "Person"},
        "Q5":  {"label": "Hong Sang-soo",    "type": "Person"},
        "Q10": {"label": "Parasite",         "type": "Film"},
        "Q11": {"label": "Memories of Murder","type": "Film"},
        "Q12": {"label": "Oldboy",           "type": "Film"},
        "Q13": {"label": "The Handmaiden",   "type": "Film"},
        "Q14": {"label": "Burning",          "type": "Film"},
        "Q15": {"label": "Poetry",           "type": "Film"},
        "Q16": {"label": "A Tale of Two Sisters","type": "Film"},
        "Q17": {"label": "I Saw the Devil",  "type": "Film"},
        "Q20": {"label": "Song Kang-ho",     "type": "Person"},
        "Q21": {"label": "Choi Min-sik",     "type": "Person"},
        "Q22": {"label": "Yoo Ah-in",        "type": "Person"},
    },
    "edges": [
        {"id": "e1",  "subject": "Q1", "predicate": "directed",   "object": "Q10"},
        {"id": "e2",  "subject": "Q1", "predicate": "directed",   "object": "Q11"},
        {"id": "e3",  "subject": "Q2", "predicate": "directed",   "object": "Q12"},
        {"id": "e4",  "subject": "Q2", "predicate": "directed",   "object": "Q13"},
        {"id": "e5",  "subject": "Q3", "predicate": "directed",   "object": "Q14"},
        {"id": "e6",  "subject": "Q3", "predicate": "directed",   "object": "Q15"},
        {"id": "e7",  "subject": "Q4", "predicate": "directed",   "object": "Q16"},
        {"id": "e8",  "subject": "Q4", "predicate": "directed",   "object": "Q17"},
        {"id": "e9",  "subject": "Q20","predicate": "starred_in", "object": "Q10"},
        {"id": "e10", "subject": "Q20","predicate": "starred_in", "object": "Q11"},
        {"id": "e11", "subject": "Q21","predicate": "starred_in", "object": "Q12"},
        {"id": "e12", "subject": "Q22","predicate": "starred_in", "object": "Q14"},
        {"id": "e13", "subject": "Q1", "predicate": "wrote",      "object": "Q10"},
        {"id": "e14", "subject": "Q3", "predicate": "wrote",      "object": "Q14"},
    ],
}


def clean_kg() -> dict[str, Any]:
    """Return a deep copy of the canonical clean KG."""
    return copy.deepcopy(CLEAN_KG)


def kg_size(kg: dict[str, Any]) -> tuple[int, int]:
    return len(kg["entities"]), len(kg["edges"])
