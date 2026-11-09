"""Business-key ID generation for all tables.

IDs are `<PREFIX>-<20 hex chars>`, generated client-side with uuid4. This
keeps ID generation collision-free across concurrent workers without a
shared counter (a shared "next sequence number" table would serialize writes
across every worker, which defeats the multi-worker design in section 17).

This is a deliberate departure from the brief's illustrative
`STUDY-000001`-style examples, which imply a global sequential counter. If
strictly sequential, zero-padded IDs are required later, swap this function
for one backed by a dedicated sequence table -- no other code depends on the
ID format.
"""
from __future__ import annotations

import uuid


# Was 12 hex chars (48 bits). Found live: with ~75 million raw_facts rows,
# each new fact had about a 1 in 3.7 million chance of reusing an existing
# id, and a settle run that writes many facts hit "UNIQUE constraint
# failed: raw_facts.fact_id". 20 hex chars (80 bits) makes that
# negligible at any realistic size. Existing 12-char ids stay valid --
# nothing parses the format.
_ID_HEX_CHARS = 20


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:_ID_HEX_CHARS]}"
