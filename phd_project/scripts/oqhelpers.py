import json
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import ast

from hazrisk.hazard import mafe_to_poe, poe_to_rtp

def get_hcurve_from_dstore(dstore, site_idx, imt , stat, investigation_time,
                           mafe:bool=True):
    hc_metadata = get_hc_metadata(dstore)
    imt_idx = hc_metadata["imt"].index(imt)
    stat_idx = hc_metadata["stat"].index(stat)
    y_data = np.array(dstore["hcurves-stats"][site_idx, stat_idx, imt_idx, :])
    if mafe:
        y_data = -np.log(1-y_data) / investigation_time

    x_data = np.array(get_imtls(dstore)[imt]).reshape(-1, 1)
    y_data = y_data.reshape(-1, 1)
    hc = np.hstack([x_data, y_data])
    return hc


def get_hc_metadata(dstore):
    hc_metadata = json.loads(dstore["hcurves-stats"].attrs["json"])
    return hc_metadata

def get_imtls(dstore):
    return dstore["oqparam"].hazard_imtls


def get_hcurves_from_dstore(dstore, mafe:bool=True):
    # if mafe == True then the poe from OQ is converted to a MAFE
    site_hcs = {}

    investigation_time = dstore["oqparam"].investigation_time
    hc_data = dstore["hcurves-stats"]
    hc_metadata = get_hc_metadata(dstore)
    all_imtls = get_imtls(dstore)

    # extract the hazard curves based on the index
    for site_i, s_data in enumerate(hc_data): # loop through the sites
        imt_hcs = {}
        for imt_i, imt in enumerate(hc_metadata["imt"]):
            imtls = all_imtls[imt]
            stat_hcs = {}
            for stat_i, stat in enumerate(hc_metadata["stat"]):
                # get the hc
                y_data = s_data[stat_i, imt_i, :].reshape(-1, 1)
                if mafe:
                    y_data = -np.log(1-y_data) / investigation_time
                
                hc = np.hstack([np.array(imtls).reshape(-1, 1),
                                y_data])
                stat_hcs[stat] = hc
            
            imt_hcs[imt] = stat_hcs
        site_hcs[site_i] = imt_hcs
    return site_hcs


def get_disagg_from_datastore(dstore, disagg_type="TRT_Mag_Dist_Eps", 
                              traditional=False, occurence=False):
    """
    Returns a dict. Each value corresponds to a site (key) and contains
    the dissaggreation results. 
    The return dict has the following structure:
    return_dict[site_id][imt][poe] = pd.DataFrame of disaggregation results
    """
    investigation_time = dstore["oqparam"].investigation_time
    site_disaggs = {}
    disagg_data = dstore["disagg-stats"][disagg_type]
    shape_descr = disagg_data.attrs["shape_descr"]

    disagg_bins = get_bins(dstore, disagg_type)

    # extract the hazard curves based on the index
    for site_i in range(disagg_data.shape[0]): # loop through the sites
        imt_disagg = {}
        for imt_i, imt in enumerate(disagg_bins["imt"]):
            poe_disagg = {}
            for poe_i, poe in enumerate(disagg_bins["poe"]):
                disagg_slice = _build_slice(site_i, poe_i, imt_i, shape_descr)
                df = _disagg_array_to_df(disagg_data[disagg_slice], 
                                         disagg_type, disagg_bins)
                if occurence:
                    imtl = get_imtl_from_hmap(dstore, site_i, imt_i, poe_i, 0)
                    df =  _get_occurence_disagg(df, dstore, site_i, imt, imtl,
                                                investigation_time)

                if traditional and not occurence:
                    df = _get_traditional_disagg(df, investigation_time)
                
                poe_disagg[poe] = df
            poe_disagg
            imt_disagg[imt] = poe_disagg
        site_disaggs[site_i] = imt_disagg

    return site_disaggs


def get_bins(dstore, disagg_type):
    
    bins = {"poe": np.array(dstore["oqparam"].poes_disagg),
            "imt": list(dstore["oqparam"].imtls.keys())}
    disagg_cols = disagg_type.split("_")

    for col in disagg_cols:
        if col == "TRT":
            bins[col] = [s.decode("UTF-8") for s in dstore["disagg-bins/TRT"]]
        else:
            bins[col] = _bins_from_edges(dstore[f"disagg-bins/{col}"])
    return bins


def _bins_from_edges(edges):
    edges = np.array(edges)
    return (edges[:-1] + edges[1:]) / 2.0


def get_probability(dstore, disagg_type, 
                    site_idx, imt_bin, poe_bin, trt_bin, mag_bin, 
                    dist_bin, eps_bin):
    
    # todo:: replace with get_bins
    mag_bins = _bins_from_edges(dstore["disagg-bins/Mag"])
    eps_bins = _bins_from_edges(dstore["disagg-bins/Eps"])
    dist_bins = _bins_from_edges(dstore["disagg-bins/Dist"])
    trt_bins = [s.decode("UTF-8") for s in dstore["disagg-bins/TRT"]]
    poe_bins = np.array(dstore["oqparam"].poes_disagg)
    imt_bins = list(dstore["oqparam"].imtls.keys())

    cell_idx = (site_idx,
                _get_bin_idx(trt_bin, trt_bins),
                _get_bin_idx(mag_bin, mag_bins),
                _get_bin_idx(dist_bin, dist_bins),
                _get_bin_idx(eps_bin, eps_bins),
                _get_bin_idx(imt_bin, imt_bins),
                _get_bin_idx(poe_bin, poe_bins))
    
    return dstore["disagg-stats"][disagg_type][cell_idx]
    

def _get_bin_idx(target, bins) -> int:
    bins = list(bins)
    return bins.index(target)


def _build_slice(site_id, poe_idx, imt_idx, shape_descr):
    n_dim = len(shape_descr)
    n_free = n_dim - 4
    index = (
        site_id, 
        *(slice(None) for _ in range(n_free)), 
        imt_idx, 
        poe_idx,
        slice(None)
    ) 
    return index


def _disagg_array_to_df(arr, disagg_type, disagg_bins):
    idx = np.indices(arr.shape)
    flat_indices = idx.reshape(len(arr.shape), -1).T
    flat_data = arr.ravel()[:, np.newaxis]
    combined_data = np.hstack((flat_indices, flat_data))
    cols = [s for s in disagg_type.split("_") ] + ["Z", 'P(X>x|T,m)']
    df = pd.DataFrame(combined_data, columns=cols)

    # replace the idx values with the actual data
    value_map = {}
    for bin, bin_vals in disagg_bins.items():
        if bin in disagg_type.split("_"):
            value_map[bin] = {ii:v for ii, v in enumerate(bin_vals)}

    for col, map in value_map.items():
        df[col] = df[col].map(map)

    return df


def _get_traditional_disagg(df, investigation_time):
    # calculate the traditional disaggregation results, P(m|X>x) from the 
    # the outputs of OQ. m = rupture
    df["nu_m"] = -np.log(1-df["P(X>x|T,m)"]) / investigation_time
    P_X_gt_x = 1 - np.prod(1-df["P(X>x|T,m)"])
    nu = -np.log(1-P_X_gt_x) / investigation_time
    df["P(m|X>x)"] = df["nu_m"] / nu
    return df


def _get_occurence_disagg(
        df, dstore, site_idx, imt, imtl, investigation_time, delta=0.001):
    # calculate the disaggregation results based on occurence, P(m|X=x)

    # first do the traditional disaggregation
    df = _get_traditional_disagg(df, investigation_time)
    hc = get_hcurve_from_dstore(
        dstore, site_idx, imt, "mean", investigation_time, mafe=True)
    delta_imtl = delta * imtl
    lambda_im1 = np.interp(imtl, hc[:,0], hc[:,1])              # lambda_im
    lambda_im2 = np.interp(imtl+delta_imtl, hc[:,0], hc[:,1])   # lambda_(im + delta_im)
    delta_lambda = lambda_im1 - lambda_im2
    df["P(m|X=x)"] = (df["P(m|X>x)"] * lambda_im1 - df["P(m|X>x)"] * lambda_im2) / delta_lambda

    return df

def get_imtl_from_hmap(dstore, site_idx, imt_idx, poe_idx, stat_idx=0):
    # only works if the poe has a hmap. Interpolation not supported
    return dstore["hmaps-stats"][site_idx, stat_idx, imt_idx, poe_idx]


# -----------------------------------------------------------------------------
# IML-based disaggregation (iml_disagg runs)
#
# Notebook 020 disaggregates at a single target IML per calculation (OpenQuake's
# ``iml_disagg``), so each datastore holds exactly one intensity level and no
# ``poes_disagg`` / ``hmaps-stats``. The POE-based reader above therefore does
# not apply; the helpers below read a single-IML datastore and collate a set of
# them (one per target IML) into a flat, un-grouped dict keyed by site index.
# -----------------------------------------------------------------------------

def get_iml_disagg_targets(dstore):
    """Return ``{imt: iml}``, the single target intensity level per IMT of an
    ``iml_disagg`` run. Read from ``oqparam.imtls`` (which the engine populates
    from ``iml_disagg``), falling back to ``oqparam.iml_disagg``.
    """
    oq = dstore["oqparam"]
    imtls = getattr(oq, "imtls", None)
    if imtls and all(len(np.atleast_1d(v)) >= 1 for v in imtls.values()):
        return {imt: float(np.atleast_1d(levels)[0])
                for imt, levels in imtls.items()}
    iml_disagg = getattr(oq, "iml_disagg", None)
    if iml_disagg:
        return {imt: float(np.atleast_1d(v)[0]) for imt, v in iml_disagg.items()}
    raise ValueError("could not determine the target IML(s) from the datastore")


def _get_occurence_disagg_from_hc(df, hc, imtl, investigation_time, delta=0.001):
    """Occurrence disaggregation ``P(m|X=x)`` using an externally supplied hazard
    curve ``hc`` for the slope.

    Mirrors :func:`_get_occurence_disagg` but takes ``hc`` (an ``(n_iml, 2)``
    array of ``[IML, MAFE]``) directly instead of reading it from the datastore.
    Needed for ``iml_disagg`` runs, whose own hazard curve has a single point and
    so cannot supply the gradient ``d(lambda)/d(IML)``. ``hc`` must come from a
    calculation with the *same* source model / logic tree as this disagg run
    (see :func:`assert_shared_provenance`).
    """
    df = _get_traditional_disagg(df, investigation_time)
    delta_imtl = delta * imtl
    lambda_im1 = np.interp(imtl, hc[:, 0], hc[:, 1])              # lambda_im
    lambda_im2 = np.interp(imtl + delta_imtl, hc[:, 0], hc[:, 1])  # lambda_(im+delta)
    delta_lambda = lambda_im1 - lambda_im2
    df["P(m|X=x)"] = (df["P(m|X>x)"] * lambda_im1
                      - df["P(m|X>x)"] * lambda_im2) / delta_lambda
    return df


def get_disagg_from_iml_datastore(dstore, disagg_type="TRT_Mag_Dist_Eps",
                                  traditional=True, occurence=False, hcurves=None):
    """Read a single-IML (``iml_disagg``) disaggregation datastore.

    Returns ``dict[site_id][imt] -> pd.DataFrame`` for the one target IML of the
    run. Reuses the POE-based building blocks (:func:`get_bins`,
    :func:`_build_slice`, :func:`_disagg_array_to_df`,
    :func:`_get_traditional_disagg`) but takes the poe axis at index 0 (there is a
    single level) and reads the target IML from :func:`get_iml_disagg_targets`
    rather than from a hazard map.

    For ``occurence=True`` pass ``hcurves`` = ``hcs[site_id][imt][stat] ->
    (n_iml, 2)`` full-resolution hazard curves (``[IML, MAFE]``) matching this
    run's source model; the ``"mean"`` stat is used for the slope.
    """
    investigation_time = dstore["oqparam"].investigation_time
    targets = get_iml_disagg_targets(dstore)
    disagg_data = dstore["disagg-stats"][disagg_type]
    shape_descr = disagg_data.attrs["shape_descr"]
    disagg_bins = get_bins(dstore, disagg_type)

    n_poe = disagg_data.shape[len(disagg_data.shape) - 2]
    if n_poe != 1:
        raise ValueError(
            f"expected a single intensity level for an iml_disagg run, "
            f"found {n_poe} on the poe axis")

    site_disaggs = {}
    for site_i in range(disagg_data.shape[0]):
        imt_disagg = {}
        for imt_i, imt in enumerate(disagg_bins["imt"]):
            imtl = targets[imt]
            disagg_slice = _build_slice(site_i, 0, imt_i, shape_descr)
            df = _disagg_array_to_df(disagg_data[disagg_slice],
                                     disagg_type, disagg_bins)
            if occurence:
                if hcurves is None:
                    raise ValueError(
                        "occurence=True requires `hcurves` for the hazard-curve "
                        "slope (the iml_disagg datastore has a single level)")
                hc = np.asarray(hcurves[site_i][imt]["mean"])
                df = _get_occurence_disagg_from_hc(
                    df, hc, imtl, investigation_time)
            elif traditional:
                df = _get_traditional_disagg(df, investigation_time)
            imt_disagg[imt] = df
        site_disaggs[site_i] = imt_disagg

    return site_disaggs


def collate_disagg_by_iml(calc_ids, imls, disagg_type="TRT_Mag_Dist_Eps",
                          traditional=True, occurence=True, hcurves=None,
                          reader=None):
    """Collate the single-IML disaggregations of one ``(IM, eps)`` pair into a
    flat, un-grouped dict keyed by site index.

    ``return[site_id][imt][iml] -> pd.DataFrame``, where ``iml`` is the target
    level in g keyed as ``float`` at 6 significant figures (stable across the
    full-precision floats OpenQuake round-trips).

    Parameters
    ----------
    calc_ids, imls : parallel ordered sequences of OpenQuake calc ids and their
        target IMLs (one per level) for a single ``(IM, eps)`` pair.
    hcurves : ``hcs[site_id][imt][stat]`` hazard curves matching the pair, passed
        through for the occurrence slope when ``occurence=True``.
    reader : datastore opener; defaults to
        ``openquake.commonlib.datastore.read``.
    """
    if reader is None:
        from openquake.commonlib.datastore import read as reader

    out = {}
    for calc_id, iml in zip(calc_ids, imls):
        iml_key = float(f"{float(iml):.6g}")
        dstore = reader(calc_id)
        site_disagg = get_disagg_from_iml_datastore(
            dstore, disagg_type, traditional, occurence, hcurves)
        for site_i, imt_dict in site_disagg.items():
            out.setdefault(site_i, {})
            for imt, df in imt_dict.items():
                out[site_i].setdefault(imt, {})[iml_key] = df
    return out


def _disagg_summary_stats(df):
    """Per-DataFrame TRT proportions (``"<TRT> [%]"``) plus PoE-weighted
    ``Mag_mean`` / ``Dist_mean``. Mirrors ``hazard._get_disagg_stats`` (kept here
    to avoid a circular import, as ``hazard`` imports from ``oqhelpers``).
    """
    trt_proportions = df.groupby("TRT")["P(m|X>x)"].sum()
    stats = {f"{i} [%]": np.round(trt_proportions.loc[i] * 100, 2)
             for i in trt_proportions.index}
    stats["Mag_mean"] = round((df["Mag"] * df["P(m|X>x)"]).sum(), 2)
    stats["Dist_mean"] = round((df["Dist"] * df["P(m|X>x)"]).sum(), 2)
    return stats


def get_iml_disagg_stats(disagg_data, site_metadata, hcurves, t=1, geodf=False):
    """Flat disagg-stats DataFrame for one ``(IM, eps)`` pair, one row per
    ``(site, imt, imtl)``.

    Columns: ``site_id, lat, lon, seismicity, region, imt, imtl, poe, mafe, rtp``
    plus the per-TRT ``"<TRT> [%]"`` proportions and ``Mag_mean`` / ``Dist_mean``
    from :func:`_disagg_summary_stats`.

    ``imtl`` is the target intensity level in g (the ``iml`` dict key). The hazard
    level at that IML is read from ``hcurves`` — ``hcs[site_id][imt][stat] ->
    (n_iml, 2)`` arrays of ``[IML, MAFE]`` (annual MAFE), the same curves used for
    the occurrence slope — by interpolating the ``"mean"`` curve: ``mafe`` at
    ``imtl``, then ``poe = mafe_to_poe(mafe, t)`` and ``rtp = poe_to_rtp(poe, t)``
    (hazrisk). ``t`` is the investigation time in years (default 1, i.e. annual
    PoE and ``rtp == 1/mafe``, matching the disagg runs' ``investigation_time``).

    ``(site, imt, imtl)`` combinations with **zero hazard** (``mafe <= 0``) are
    omitted: the IML is beyond the maximum ground motion permitted by the GMM's
    epsilon truncation, so the return period is infinite and the disaggregation is
    not a meaningful hazard scenario. Those IMLs stay in ``disagg_data`` and are
    listed by :func:`get_excluded_imls`.
    """
    all_stats = []
    for site_id, imt_dict in disagg_data.items():
        for imt, iml_dict in imt_dict.items():
            hc = np.asarray(hcurves[site_id][imt]["mean"])
            for imtl, df in iml_dict.items():
                mafe = float(np.interp(imtl, hc[:, 0], hc[:, 1]))
                if mafe <= 0:
                    continue  # zero hazard (truncation); excluded from stats
                poe = float(mafe_to_poe(mafe, t))
                rtp = float(poe_to_rtp(poe, t))
                row = {"site_id": site_id,
                       "lat": site_metadata.loc[site_id, "lat"],
                       "lon": site_metadata.loc[site_id, "lon"],
                       "seismicity": site_metadata.loc[site_id, "seismicity"],
                       "region": site_metadata.loc[site_id, "region"],
                       "imt": imt,
                       "imtl": imtl,
                       "poe": poe,
                       "mafe": mafe,
                       "rtp": rtp}
                all_stats.append(row | _disagg_summary_stats(df))

    df = pd.DataFrame.from_records(all_stats)
    if geodf:
        import geopandas as gpd
        df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    return df


def get_excluded_imls(disagg_data, hcurves):
    """List the zero-hazard IMLs dropped from the stats by
    :func:`get_iml_disagg_stats`.

    Returns ``{site_id: [imls]}`` (imls sorted, in g) for every ``(site, imt,
    iml)`` whose mean hazard curve gives ``mafe <= 0`` — i.e. an intensity level
    beyond the epsilon-truncated maximum ground motion, with an infinite return
    period. Sites with no excluded IMLs are omitted. These IMLs remain present in
    ``disagg_data``; only the stats rows are dropped.
    """
    excluded = {}
    for site_id, imt_dict in disagg_data.items():
        for imt, iml_dict in imt_dict.items():
            hc = np.asarray(hcurves[site_id][imt]["mean"])
            for imtl in iml_dict:
                mafe = float(np.interp(imtl, hc[:, 0], hc[:, 1]))
                if mafe <= 0:
                    excluded.setdefault(site_id, set()).add(imtl)
    return {s: sorted(imls) for s, imls in excluded.items()}


def assert_shared_provenance(disagg_manifest, psha_manifest, im, eps,
                             keys=("gmpe_lt", "ssc_lt", "site_model",
                                   "source_models_digest")):
    """Assert that the PSHA calculation supplying the hazard curves for
    ``(im, eps)`` shares source-model provenance with the disagg runs.

    The occurrence slope is taken from the PSHA hazard curves, so those curves
    must come from the same source model / logic tree / site model as the disagg
    calculations. Compares the input hashes of ``AvgSA_{im}_psha_eps{eps}``
    against every ``AvgSA_{im}_disagg_eps{eps}_*`` entry and raises on any
    mismatch. ``im`` is e.g. ``"03"``; ``eps`` is e.g. ``3``.
    """
    psha_name = f"AvgSA_{im}_psha_eps{eps}"
    if psha_name not in psha_manifest:
        raise KeyError(f"{psha_name} not in the PSHA manifest")
    psha_inputs = psha_manifest[psha_name]["inputs"]

    prefix = f"AvgSA_{im}_disagg_eps{eps}_"
    disagg_names = [k for k in disagg_manifest if k.startswith(prefix)]
    if not disagg_names:
        raise KeyError(f"no disagg manifest entries matching {prefix}*")

    for name in disagg_names:
        di = disagg_manifest[name]["inputs"]
        for key in keys:
            dh = di.get(key, {}).get("hash")
            ph = psha_inputs.get(key, {}).get("hash")
            if dh != ph:
                raise ValueError(
                    f"provenance mismatch between {name} and {psha_name} on "
                    f"'{key}': disagg [{dh}] != psha [{ph}]. The hazard curves "
                    f"cannot supply the occurrence slope for this pair.")
    return True


def parse_value(val_str):
    """Converts string values to appropriate Python types."""
    val_str = val_str.strip()
    
    # Handle Strings: "value" or 'value'
    if (val_str.startswith('"') and val_str.endswith('"')) or \
       (val_str.startswith("'") and val_str.endswith("'")):
        return val_str[1:-1]
    
    # Handle Lists: [1, 2, 3]
    if val_str.startswith('[') and val_str.endswith(']'):
        try:
            return ast.literal_eval(val_str)
        except (ValueError, SyntaxError):
            return val_str
            
    # Handle Numbers: 1.0, -2.85
    try:
        return float(val_str)
    except ValueError:
        return val_str

def parse_nrml_logic_tree(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    results = {}

    # Iterate through each tectonic region branch set
    for branch_set in root.findall('.//{*}logicTreeBranchSet'):
        region_type = branch_set.get('applyToTectonicRegionType')
        
        if region_type not in results:
            results[region_type] = {}
        
        for branch in branch_set.findall('{*}logicTreeBranch'):
            branch_id = branch.get('branchID')
            
            # Extract and convert weight
            weight_elem = branch.find('{*}uncertaintyWeight')
            weight = float(weight_elem.text.strip()) if weight_elem is not None else 1.0
            
            model_elem = branch.find('{*}uncertaintyModel')
            params = {}
            if model_elem is not None and model_elem.text:
                model_text = model_elem.text.strip()
                for line in model_text.split('\n'):
                    line = line.strip()
                    # Skip the [GenericGmpeAvgSA] or [ModelName] headers
                    if not line or (line.startswith('[') and ']' in line and '=' not in line):
                        continue
                    
                    if '=' in line:
                        k, v = line.split('=', 1)
                        params[k.strip()] = parse_value(v.strip())
                    else:
                        # Fallback for simple model names like 'LanzanoLuzi2019'
                        params['model_name'] = line
            
            results[region_type][branch_id] = {
                'weight': weight,
                'params': params
            }
            
    return results


if __name__ == "__main__":
    from phd_project.config.config import load_config
    cfg = load_config()
    fp = cfg["hazard_models"]["eshm20_AvgSA"] / "gmpe_logic_tree_AvgSA_0to3_median_branch.xml"
    output = parse_nrml_logic_tree(file_path=fp)
    pass