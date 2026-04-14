"""
prompts/store.py
================
Central registry of every prompt version the pipeline has tried.

Design principles
-----------------
  - All prompts are stored here, versioned sequentially (v1, v2, v3, …).
  - The optimiser appends new versions; it never modifies existing ones.
    This gives a complete, auditable history of what was tried and why.
  - The classifier always loads whatever version is marked as "current".
  - Gold labels are NEVER stored here — this is a prompt store, not a dataset.

Prompt structure
----------------
  system : str  — Sent as the "system" role message.
                  Contains the relevancy criteria and output format rules.
  user   : str  — Sent as the "user" role message.
                  Template with {query} and {document} placeholders.
  notes  : str  — Human-readable explanation of what changed in this version
                  and why (written by the optimiser for each new version).

Starting prompts
----------------
  v1 — Intentionally minimal baseline.  We start vague so the optimiser has
       clear room to improve and we can observe the full optimisation journey.
  v2 — Adds explicit positive/negative criteria; a typical first manual step.

The pipeline begins at v1 by default.  You can change start_version in main.py
to skip ahead if you already have a reasonable starting prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Prompt:
    """One versioned snapshot of the classification prompt."""
    version: str          # e.g. "v1", "v2", "v3"
    system:  str          # system role content
    user:    str          # user role template — use {query} and {document}
    notes:   str = ""     # explanation written by optimiser (empty for manual versions)

    # Timestamp recorded when the prompt is created; useful for audit logs.
    created: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Built-in starting prompts ─────────────────────────────────────────────────

# v1 — Minimal baseline.
# Problems we expect: vague role with no criteria — the model must guess what
# "relevant" means in a legal document review context, leading to inconsistent
# classifications and likely low precision or recall.
_V1 = Prompt(
    version="v1",
    system=(
        "You are a document review attorney. "
        "For each document, decide if it is relevant to the legal matter."
    ),
    user=(
        "Query: {query}\n\n"
        "Document: {document}\n\n"
        'Reply with JSON only:\n'
        '{{"label": "relevant" or "not_relevant", '
        '"confidence": 0.0-1.0, '
        '"reason": "one sentence explaining your decision"}}'
    ),
    notes="Baseline — intentionally minimal to establish a clear starting point.",
)

# v2 — Adds explicit relevant / not-relevant criteria from the review protocol.
# This is what a thoughtful review attorney would specify as coding guidelines.
# Covers the six relevance categories (agreements, communications, invoices,
# ERCOT approval, Odessa Plant, and project references) plus explicit
# not-relevant rules to reduce false positives on tangential energy docs.
_V2 = Prompt(
    version="v2",
    system=(
        "You are a document review attorney classifying documents for "
        "relevancy in the matter of Luminant / Vistra vs. Cipher.\n\n"
        "Classify a document as RELEVANT if it meets ANY of these criteria:\n"
        "  1. Agreements & Negotiations — Any agreements or negotiations "
        "between Luminant and Cipher, including the Lease Agreement between "
        "Cipher and La Frontera and the Purchase and Sale Agreement between "
        "Vistra and Cipher.\n"
        "  2. Communications — Any communications between Cipher and "
        "Luminant and/or Vistra regarding the agreements, the data mining "
        "center, the Odessa Plant, or the Substation, including internal "
        "communications or those with third parties.\n"
        "  3. Invoices & Financial Records — Invoices exchanged between "
        "the parties, or evidence of payments or credits from Luminant "
        "to Cipher.\n"
        "  4. ERCOT Approval — Documentation of ERCOT's approval of "
        "energization and use of the Substation.\n"
        "  5. Odessa Plant — Documents related to the Odessa Plant, "
        "specifically those concerning the Plant providing power to "
        "Cipher's facility.\n"
        "  6. Project References — Any documents referring to the project "
        'as "Bitfury" or "Cipher", including misspellings such as '
        '"Cypher".\n\n'
        "Classify a document as NOT_RELEVANT if it:\n"
        "  - Relates to energy or power operations but has no connection "
        "to Cipher, Luminant, Vistra, the Odessa Plant, or the "
        "Substation.\n"
        "  - Mentions general ERCOT matters unrelated to the Substation "
        "energization approval.\n"
        "  - Is a routine internal document (HR, IT, general admin) with "
        "no reference to the parties or subject matter above.\n\n"
        "Reply with JSON only — no other text."
    ),
    user=(
        "Query: {query}\n\n"
        "Document: {document}\n\n"
        'Reply with JSON only:\n'
        '{{"label": "relevant" or "not_relevant", '
        '"confidence": 0.0-1.0, '
        '"reason": "one sentence explaining your decision"}}'
    ),
    notes=(
        "Added explicit six-category relevance criteria from the review "
        "protocol and not-relevant rules to reduce false positives on "
        "tangential energy/ERCOT documents."
    ),
)


# ── Registry state ────────────────────────────────────────────────────────────

# All known versions, keyed by version string.
# We use a plain dict (insertion-ordered in Python 3.7+) so iteration order
# reflects creation order, which is useful for the summary table.
_versions: dict[str, Prompt] = {
    _V1.version: _V1,
    _V2.version: _V2,
}

# The version the classifier will use on the next run.
# main.py sets this at startup and updates it after each optimiser call.
_current: str = "v1"


# ── Brace escaping ────────────────────────────────────────────────────────────

def _escape_braces(template: str) -> str:
    """Escape braces in a user template, preserving {query} and {document}.

    The optimiser writes JSON examples with literal braces like
    {"label": ...} but Python's str.format() treats those as placeholders.
    We double all braces first, then restore the two real placeholders.
    """
    escaped = template.replace("{", "{{").replace("}", "}}")
    escaped = escaped.replace("{{query}}", "{query}")
    escaped = escaped.replace("{{document}}", "{document}")
    return escaped


# ── Public API ────────────────────────────────────────────────────────────────

def current() -> Prompt:
    """Return the prompt version currently marked as active."""
    return _versions[_current]


def get(version: str) -> Prompt:
    """Return a specific version by its version string (e.g. 'v2')."""
    return _versions[version]


def set_current(version: str) -> None:
    """Make a version the active one.  Called by main.py after optimisation."""
    global _current
    if version not in _versions:
        raise KeyError(f"Unknown prompt version: {version!r}")
    _current = version


def register(system: str, user: str, notes: str) -> Prompt:
    """
    Create and store a new prompt version.

    Called by the optimiser agent after it generates an improved prompt.
    The version number is assigned automatically (next integer in sequence).

    Parameters
    ----------
    system : The new system prompt content.
    user   : The new user template (must contain {query} and {document}).
    notes  : The optimiser's explanation of what changed and why.
             This becomes the audit trail for this version.

    Returns
    -------
    The newly created Prompt object.
    """
    # The optimiser doesn't know about Python's str.format() escaping, so
    # it writes JSON examples with single braces like {"label": ...}.
    # We escape all braces except the {query} and {document} placeholders
    # to prevent KeyError when .format() is called in the classifier.
    user = _escape_braces(user)

    # Derive next version number from the current count of registered versions.
    next_number = len(_versions) + 1
    new_prompt = Prompt(
        version=f"v{next_number}",
        system=system,
        user=user,
        notes=notes,
    )
    _versions[new_prompt.version] = new_prompt
    return new_prompt


def all_versions() -> list[Prompt]:
    """Return all registered versions in creation order."""
    return list(_versions.values())