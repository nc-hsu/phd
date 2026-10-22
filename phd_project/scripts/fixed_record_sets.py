"""Fixed (FX) record sets: the record-tag convention shared by every FX analysis.

A fixed set is used unchanged for every design group and stripe, so every analysis that
uses it - the group IDAs (nb 052/053) and the group MSAs (nb 054/061) alike - refers to a
record by its **tag**: the record's position in the SORTED listing of the set's record
JSONs (``record_sets.fx.{fx}.record_glob`` in config.yaml; ``config.fx_record_files``
builds the listing). That single convention is what makes the two arms comparable
record-by-record and lets one set of bootstrap resamples (nb 070) apply to either, so it
lives here rather than inside any one analysis module.

Example (FEMA P695 far-field, glob ``fema_p695_*.json``): tag ``"0"`` is
``fema_p695_120111``, tag ``"21"`` the last of the 22 sorted files.
"""

from collections.abc import Sequence
from pathlib import Path


def record_stems(record_files: Sequence[Path | str]) -> list[str]:
    """The record stems, in TAG order, of an FX set's record files.

    Pass ``config.fx_record_files(cfg, fx)`` (already sorted). A record's position in the
    returned list *is* its record tag: the IDA / MSA config templates key the runs by
    ``{str(ii): record for ii, record in enumerate(...)}`` over the same sorted listing, and
    the tags are the column labels of the per-record collapse-IML / collapse-flag tables.
    The order is checked rather than assumed, because an unsorted list would silently
    attach every result to the wrong record.
    """
    stems = [Path(p).stem for p in record_files]
    names = [Path(p).name for p in record_files]
    if names != sorted(names):
        raise ValueError("record files must be in sorted order (config.fx_record_files)")
    if not stems:
        raise FileNotFoundError("no record files given - is the records folder right?")
    return stems


def record_tag_to_column(record_tags: Sequence[str]) -> dict[str, str]:
    """Map record stem -> collapse-IML table column label (``"0"`` ... ``"21"``)."""
    return {tag: str(i) for i, tag in enumerate(record_tags)}
