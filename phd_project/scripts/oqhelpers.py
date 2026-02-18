import json
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import ast

def get_hcurve_from_dstore(dstore, site, im , stat):
    # TODO
    ...

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
                    df =  _get_occurence_disagg(df)

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


def _get_occurence_disagg(df, investigation_time, imtl, delta=0.001):
    # calculate the disaggregation results based on occurence, P(m|X=x)

    # first do the traditional disaggregation
    df = _get_traditional_disagg(df, investigation_time)
    delta_im = delta * imtl
    lambda_im1 = ... # lambda_im
    lambda_im2 = ... # lambda_(im + delta_im)
    delta_lambda = lambda_im1 - lambda_im2
    df["P(m|X=x)"] = (df["P(m|X>x)"] * lambda_im1 - df["P(m|X>x)"] * lambda_im2) / delta_lambda


def get_imtl_from_hmap(dstore, site_idx, imt_idx, poe_idx, stat_idx=0):
    # only works if the poe has a hmap. Interpolation not supported
    return dstore["hmaps-stats"][site_idx, stat_idx, imt_idx, poe_idx]


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