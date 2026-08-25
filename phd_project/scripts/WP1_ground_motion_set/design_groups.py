"""Sites that share a structural design.

Notebook 011 designs a CBF per (site, storey count) from the site's ``S_alpha,475``
alone. The section catalogue is discrete, so many sites end up with a byte-identical
design and therefore an identical model - one analysis covers the whole group.

Notebook 051 works those groups out and writes them to
``unique_structural_designs.csv`` / ``.json``; this module reads them back.
"""

import json
from pathlib import Path

import pandas as pd

from phd_project.config import config


def load_design_groups(cfg: dict | None = None) -> pd.DataFrame:
    """The (site, storey) -> design-group table written by notebook 051.

    Columns: ``storeys``, ``site``, ``tag``, ``group_id``, ``fingerprint``,
    ``representative_site``, ``representative_tag``, ``is_representative``,
    ``n_sites_in_group``. Sites sharing a ``group_id`` (within a storey count) have
    identical structural designs.
    """
    cfg = cfg or config.load_config()
    path = Path(cfg["proc_data"]["unique_structural_designs_csv"])
    if not path.is_file():
        raise FileNotFoundError(
            f"{path.name} not found - run notebook 051 first ({path})")
    return pd.read_csv(path)


def load_design_groups_json(cfg: dict | None = None) -> dict:
    """The nested group description written by notebook 051.

    ``{"groups": {"3": [{group_id, fingerprint, representative_site, sites, ...}], ...}}``
    - the same grouping as :func:`load_design_groups`, with the member lists and the
    group's sections kept together. Use the dataframe for joins, this for reporting.
    """
    cfg = cfg or config.load_config()
    path = Path(cfg["proc_data"]["unique_structural_designs"])
    if not path.is_file():
        raise FileNotFoundError(
            f"{path.name} not found - run notebook 051 first ({path})")
    with open(path) as f:
        return json.load(f)


def representative_sites(storeys: int, cfg: dict | None = None,
                         groups_df: pd.DataFrame | None = None) -> list[int]:
    """The site indices to actually analyse for ``storeys`` - one per design group."""
    df = load_design_groups(cfg) if groups_df is None else groups_df
    mask = (df["storeys"] == storeys) & df["is_representative"]
    return sorted(df.loc[mask, "site"].astype(int))


def representative_of(site: int, storeys: int, cfg: dict | None = None,
                      groups_df: pd.DataFrame | None = None) -> int:
    """The site whose analysis results apply to ``site`` (itself, if it is the rep)."""
    df = load_design_groups(cfg) if groups_df is None else groups_df
    rows = df[(df["storeys"] == storeys) & (df["site"] == site)]
    if rows.empty:
        raise KeyError(f"site {site} [{storeys}s] is not in the design-group table")
    return int(rows.iloc[0]["representative_site"])
