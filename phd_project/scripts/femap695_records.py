"""The FEMA P695 far-field record set and its record-tag convention.

The far-field set is fixed, so every analysis that uses it - the group IDAs (nb 052/053)
and the group MSAs (nb 054/061) alike - refers to a record by its **tag**: the record's
position in the sorted ``fema_p695_*.json`` listing. That single convention is what makes
the two arms comparable record-by-record and lets one set of bootstrap resamples (nb 070)
apply to either, so it lives here rather than inside any one analysis module.
"""

from collections.abc import Sequence
from pathlib import Path


def femap695_record_tags(records_dir: Path | str) -> list[str]:
    """Return the FEMA P695 record stems in the order the IDA indexed them.

    A record's position in this list *is* its IDA record tag: notebooks 011 and 052 set
    ``FEMAP695_RECORDS`` to the sorted ``fema_p695_*.json`` listing, and the IDA config
    template keys the run by ``{str(ii): record for ii, record in enumerate(...)}``. So
    tag ``"0"`` is ``fema_p695_120111``, and the tags are the column labels of the
    per-record collapse-IML tables.
    """
    return [p.stem for p in sorted(Path(records_dir).glob("fema_p695_*.json"))]


def record_tag_to_column(record_tags: Sequence[str]) -> dict[str, str]:
    """Map record stem -> collapse-IML table column label (``"0"`` ... ``"21"``)."""
    return {tag: str(i) for i, tag in enumerate(record_tags)}
