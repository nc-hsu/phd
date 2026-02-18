import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import geopandas as gpd
from phd_project.scripts.oqhelpers import (
    get_hc_metadata, get_imtls, get_hcurves_from_dstore,
    get_disagg_from_datastore)


def group_hazard_curves(groups: list[tuple], metadata, dstore,
                        calculate_mean_curves:bool=True):

    """
    Groups the hazard curves stored in an openquake datastore according to
    groups and metadata.
    The mean curves for each groupd are calculated
    Returns a dictionary with the following levels
    return_dict[group][site][imt][stat] = hazard curve np.ndarray shape (:, 2)
    """

    hc_metadata = get_hc_metadata(dstore)
    imtls = get_imtls(dstore)
    hcs = get_hcurves_from_dstore(dstore)

    grouped_hcs = {}
    for s, r in groups:
        mask = get_mask(["seismicity", "region"], [s, r], metadata)
        sites = metadata[mask]
        
        group_hcs = {idx : hcs[idx] for idx in sites.index}
        temp = {}

        if not calculate_mean_curves:
            grouped_hcs[(s, r)] = group_hcs
            continue

        for imt in imtls.keys():
            temp[imt] = {}
            for stat in hc_metadata["stat"]:
                mean_curve = np.mean(
                    np.vstack([shc[imt][stat][:,1] 
                               for shc in group_hcs.values()]), axis=0)
                temp[imt][stat] = np.vstack([np.array(imtls[imt]),
                                                    mean_curve]).T

        group_hcs["mean_site"] = temp
        grouped_hcs[(s, r)] = group_hcs
    return grouped_hcs
        


def get_mask(keys: list, targets: list, df: pd.DataFrame):
    """
    Returns a mask where all specified columns match 
    their respective target values.

    param: cols list: names of columns being considered
    param: targets list: list of target values for each category
    param: df pd.Dataframe: the df we create the mask from
    """
    # Create a list of boolean Series for each condition
    conditions = [df[c] == t for c, t in zip(keys, targets)]
    
    # Use reduce or concat to find where all conditions are True
    mask = pd.concat(conditions, axis=1).all(axis=1)
    
    return mask


def group_disagg_data(groups: list[tuple], metadata, dstore,
                      disagg_type, traditional, occurence=False):
    """
    Groups the dissagregation data stored in an openquake datastore according to
    groups and metadata.
    Returns a dictionary with the following levels
    return_dict[group][site][imt][poe] = pd.DataFrame representing the
    disaggregation results.
    """

    disagg_data = get_disagg_from_datastore(dstore, disagg_type, traditional, occurence)

    grouped_data = {}
    for s, r in groups:
        mask = get_mask(["seismicity", "region"], [s, r], metadata)
        sites = metadata[mask]
        group = {idx : disagg_data[idx] for idx in sites.index}
        grouped_data[(s, r)] = group
    return grouped_data


def _get_disagg_stats(df):

    trt_proportions = df.groupby("TRT")["P(m|X>x)"].sum()
    stats = {f"{i} [%]": np.round(trt_proportions.loc[i] * 100, 2) 
             for i in trt_proportions.index}
    stats["Mag_mean"] = round((df["Mag"] * df["P(m|X>x)"]).sum(),2)
    stats["Dist_mean"] = round((df["Dist"] * df["P(m|X>x)"]).sum(),2)
    # stats["Eps_mean"] = round((df["Eps"] * df["P(m|X>x)"]).sum(),2)
    return stats


def get_disagg_stats_from_groups(groups_disagg, site_metadata, hmaps,
                                 stat_idx=0, geodf:bool=False):
    all_stats = []
    for (s, r), vi in groups_disagg.items():
        for site_idx, (site_id, vj) in enumerate(vi.items()):
            for imt_idx, (imt, vk) in enumerate(vj.items()):
                for poe_idx, (poe, vl) in enumerate(vk.items()):
                    stats = {"site_id": site_id,
                             "lat": site_metadata.loc[site_idx, "lat"],
                             "lon": site_metadata.loc[site_idx, "lon"],
                             "seismicity": s,
                             "region": r,
                             "imt": imt,
                             "poe": poe,
                             "imtl": hmaps[site_idx, stat_idx, imt_idx, poe_idx]
                    } 
                    all_stats.append(stats | _get_disagg_stats(vl))

    df = pd.DataFrame.from_records(all_stats)
    if geodf:
        df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    return df












# def plot_disaggregation_3d(ax, df, catx, caty, catz, colour_map, dx, dy):
#     agg_df = df.groupby([catx, caty, catz])['P(m|X>x)'].sum().reset_index()

#     # Get unique sorted values for axes
#     x_vals = np.array(sorted(agg_df[catx].unique()))
#     y_vals = np.array(sorted(agg_df[caty].unique()))
#     catz_vals = np.array(sorted(agg_df[catz].unique()))
    
#     # Create mappings for grid placement
#     # x_map = {val: i for i, val in enumerate(x_vals)}
#     # y_map = {val: i for i, val in enumerate(y_vals)} 

#     # Track the "bottom" height for stacking
#     bottom = np.zeros((len(x_vals), len(y_vals)))
    
#     # Standard color map for Epsilon bins
#     colors = colour_map  

#     # 2. Plot each Epsilon bin as a layer
#     for i, catz_val in enumerate(catz_vals):
#         # subset = agg_df[agg_df[catz] == catz_val]
        
#         # Height of current epsilon layer
#         dz = agg_df[agg_df[catz] == catz_val].pivot_table(index=catx, 
#                 columns=caty, values='P(m|X>x)', aggfunc='sum')
#         dz = dz.reindex(index=x_vals, columns=y_vals).fillna(0).values
        
#         # Flatten for bar3d
#         xpos, ypos = np.meshgrid(x_vals - dx/2, y_vals - dy/2, indexing='ij')
#         xpos = xpos.ravel()
#         ypos = ypos.ravel()
#         zpos = bottom.ravel()
#         dz_flat = dz.ravel()
        
#         # Draw the 3D bars for this Eps slice
#         ax.bar3d(xpos, ypos, zpos, dx, dy, dz_flat, 
#                  color=colors[i], label=f'{catz}: {catz_val}', alpha=1)
        
#         # Update bottom for the next Eps layer
#         bottom += dz 

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     from openquake.commonlib.datastore import read
#     from phd_project.scripts.oqhelpers import get_bins
    
    
#     # load the site model and the sites
#     sel_sites = pd.read_csv(cfg["results"]["selected_sites_csv"])
#     site_model = pd.read_csv(cfg["hazard_models"]["eshm20_AvgSA_site_model_all"])
#     site_metadata = pd.concat([sel_sites, site_model], axis=1).T.drop_duplicates().T
#     site_metadata.columns

#     groups = list(site_metadata.groupby(["seismicity", "region"]).size().index)

#     calc_id = 16
#     dstore = read(calc_id)
#     disagg_type = 'TRT_Mag_Dist_Eps'
#     bins = get_bins(dstore, disagg_type)
#     groups_disagg = group_disagg_data(groups, site_metadata, dstore, 
#                                   disagg_type, traditional=True)
#     hmaps = dstore["hmaps-stats"]
#     disagg_stats = get_disagg_stats_from_groups(groups_disagg, site_metadata,
#                                                 hmaps, geodf=False)
    
    # # Matplotlib
    # # Plot a disaggregation plot 1x2.
    # # Left side shows colours for TRT and the right for Eps
    # cmap = plt.get_cmap("jet", len(bins["Eps"]))
    # bounds = np.linspace(-3, 3, 7)
    # # cmaplist = [cmap(i) for i in range(cmap.N)]
    # # # create the new map
    # # cmap = mpl.colors.LinearSegmentedColormap.from_list(
    # # 'Custom cmap', cmaplist, cmap.N)

    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # df = groups_disagg[("lowmod", 0)][0]["AvgSA"][0.02]
    # # plot_disaggregation_3d(ax, df, "Mag", "Dist", "Eps", colour_map, dx=0.5, dy=20)

    # elev = 30
    # azim = 60
    # catx = "Mag"
    # caty = "Dist"
    # catz = "Eps"
    # dx = 0.1
    # dy = 10
    # width_sf = 1.0

    # agg_df = df.groupby([catx, caty, catz])['P(m|X>x)'].sum().reset_index()
    # agg_df["bar_top"] = agg_df.groupby([catx, caty])['P(m|X>x)'].cumsum()
    # agg_df["bar_bot"] = agg_df["bar_top"] - agg_df['P(m|X>x)']

    # # x_range = df[catx].max() - df[catx].min()
    # # y_range = df[caty].max() - df[caty].min()
    # # ax.set_box_aspect((x_range, y_range, (x_range + y_range) / 4))

    # # 2. Plot each Epsilon bin as a layer
    # catz_vals = sorted(agg_df[catz].unique())
    # for ii, catz_val in enumerate(catz_vals):
        
    #     subset = agg_df[agg_df[catz] == catz_val]
    #     # Draw the 3D bars for this Eps slice
    #     ax.bar3d(subset[catx]-dx/2, subset[caty]-dy/2, subset["bar_bot"], 
    #              dx * width_sf, dy * width_sf, subset["P(m|X>x)"], 
    #              color=cmap(ii), label=f'{catz}: {catz_val}', shade=False,
    #              edgecolor="k", linewidth=0.25, alpha=0.8)
    #     ax.view_init(elev=elev, azim=azim, roll=0)
    # plt.show()

    # # ## PyVista
    # # catx = "Mag"
    # # caty = "Dist"
    # # catz = "Eps"
    # # dx = 0.1
    # # dy = 10
    # # width_sf = 1.0

    # # poe_df = groups_disagg[("lowmod", 0)][0]["AvgSA"][0.02]
    # # df = poe_df.groupby([catx, caty, catz])['P(m|X>x)'].sum().reset_index()
    # # df["bar_top"] = df.groupby([catx, caty])['P(m|X>x)'].cumsum()
    # # df["bar_bot"] = df["bar_top"] - df['P(m|X>x)']

    # # # Define a colormap for Epsilon
    # # catz_vals = np.sort(df[catz].unique())
    # # colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

    # # # Axes and ranges
    # # x_min, x_max = 4.0, 10.0
    # # y_min, y_max = 0, 500
    # # z_min, z_max = 0, df["bar_top"].max()

    # # # 2. Initialize Plotter
    # # plotter = pv.Plotter()
    
    # # # We want 1 unit of Magnitude to look like 'scale' units of Distance
    # # # If dy=10 and dx=0.1, we need x_scale to be 100 to make the base square
    # # desired_x_size = dy  # We want the x-width to visually match the y-length
    # # x_scale_factor = desired_x_size / dx 

    # # # For Z, we want the max bar height to be visually significant (e.g., 1/4 of Y range)
    # # z_scale_factor = (y_max / 4) / df["bar_top"].max()

    # # # 3. Create Meshes
    # # for i, catz_val in enumerate(catz_vals):
    # #     subset = df[df[catz] == catz_val]
        
    # #     # We can combine all bars of the same Epsilon into one mesh 
    # #     # for better performance
    # #     eps_meshes = []
    # #     for _, row in subset.iterrows():
    # #         # Create a box at the specific location
    # #         # PyVista Box bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
    # #         box = pv.Box(bounds=(
    # #             row[catx] - dx/2, row[catx] + dx/2,
    # #             row[caty] - dy/2, row[caty] + dy/2,
    # #             row["bar_bot"], row["bar_top"]
    # #         ))
    # #         eps_meshes.append(box)
        
    # #     # Merge all boxes for this Epsilon into a single multiblock or polydata
    # #     combined = pv.merge(eps_meshes)
        
    # #     # Add to plotter
    # #     actor = plotter.add_mesh(
    # #         combined, 
    # #         color=colors[i % len(colors)], 
    # #         label=f"{catz} {catz_val}",
    # #         smooth_shading=False,
    # #         show_edges=True,  # This gives you the clean black outlines
    # #         edge_color='black'
    # #     )

    # #     # APPLY THE SCALE TO THE ACTOR
    # #     # This stretches the Magnitude and Probability visually
    # #     actor.scale = [x_scale_factor, 1.0, z_scale_factor]
    

    # # # # Create manual tick arrays in "Engineering Units"
    # # # x_ticks_data = np.arange(x_min, x_max + 0.5, 0.5)
    # # # y_ticks_data = np.arange(y_min, y_max + 50, 50)

    # # # 4. Final Formatting
    # # # This defines the visual box where the grid lives
    # # scaled_bounds = (
    # #         x_min * x_scale, x_max * x_scale,
    # #         y_min, y_max,
    # #         z_min * z_scale, z_max * z_scale
    # #     )

    # # grid_actor = plotter.show_grid(
    # #     bounds=scaled_bounds,
    # #     xtitle="Magnitude (Mw)",
    # #     ytitle="Distance (km)",
    # #     ztitle="Contribution",
    # #     # Pass scaled world positions for tick placement
    # #     # xticks=x_ticks_data * x_scale,
    # #     # yticks=y_ticks_data * y_scale,
    # #     # Pass strings for the actual labels shown
    # #     # xticklabels=[f"{v:.1f}" for v in x_ticks_data],
    # #     # yticklabels=[f"{int(v)}" for v in y_ticks_data],
    # #     color='black',
    # #     location='outer',
    # #     padding=0.05
    # # )

    # # plotter.add_legend(bcolor='white')
    # # plotter.set_background("white")
    # # plotter.show()


    # pass
