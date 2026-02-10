import json
import numpy as np

def get_hcurve_from_dstore(dstore, site, im , stat):
    # TODO
    ...

def get_hc_metadata(dstore):
    hc_metadata = json.loads(dstore["hcurves-stats"].attrs["json"])
    return hc_metadata

def get_imtls(dstore):
    return dstore["oqparam"].hazard_imtls


def get_hcurves_from_dstore(dstore):
    site_hcs = {}

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
                hc = np.hstack([np.array(imtls).reshape(-1, 1),
                                s_data[stat_i, imt_i, :].reshape(-1, 1)])
                stat_hcs[stat] = hc
            
            imt_hcs[imt] = stat_hcs
        site_hcs[site_i] = imt_hcs
    return site_hcs


def get_disagg_from_datastore(dstore, disagg_type="TRT_Mag_Dist_Eps", 
                              traditional=False):
    """
    Returns a dict. Each value corresponds to a site (key) and contains
    the dissaggreation results. 
    The return dict has the following structure:
    return_dict[site_id][imt][poe] = pd.DataFrame of disaggregation results
    """
    site_disaggs = {}
    disagg_data = dstore["disagg-stats"][disagg_type]
    shape_descr = disagg_data.attrs["shape_descr"]

    disagg_bins = get_bins(dstore, disagg_type)

    # TODO:: get the iml from the datastore / hazard maps as well

    # extract the hazard curves based on the index
    for site_i in range(disagg_data.shape[0]): # loop through the sites
        imt_disagg = {}
        for imt_i, imt in enumerate(disagg_bins["imt"]):
            poe_disagg = {}
            for poe_i, poe in enumerate(disagg_bins["poe"]):
                disagg_slice = _build_slice(site_i, poe_i, imt_i, shape_descr)
                poe_disagg[poe] = disagg_data[disagg_slice]
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
