import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import hazrisk.risk as risk
from pathlib import Path
import phd_project.scripts.fragility_data_models as fm
from matplotlib.transforms import blended_transform_factory

############      FRAGILITY CORRECTION     ###############
def fragility_parameters(model_registry, model, frag_type="c_avg"):
    """ retrieves the fragility parameters for all sites for the provided fragility type:
    frag_type == c_avg  --> ida fragility using the average collapse im
    frag_type == a      --> msa site specific (from the nltha)"""
    spec = model.spec
    # to_rows returns (replicates, estimates); the [1] picks the point estimates (log scale).
    theta_ida = np.exp(model_registry.theta_data.to_rows(frag_type)[1])
    beta_ida  = np.exp(model_registry.beta_data.to_rows(frag_type, corrected=spec.corrected)[1])
    df = pd.concat([theta_ida, beta_ida], axis=1)
    return df


def estimate_of_msa_from_ida(model_registry, model, site, n_storeys) -> tuple[float, float]:
    """ returns the estimated theta and beta for the msa-ss based on the
    ida results and the predicted correction factor.
    The ida results are called up from the model directly."""
    theta_ida, beta_ida = fragility_parameters(model_registry, model).loc[(site, n_storeys), :].values
    theta_cf, beta_cf = model_fit_estimates(model , site, n_storeys)

    theta_msa_est = theta_ida * np.exp(theta_cf)
    beta_msa_est = beta_ida * np.exp(beta_cf)
    return theta_msa_est, beta_msa_est


def msa_estimate_std_errors(model_registry, model, site, n_storeys, ida_tag):
    """ returns the standard errors on ln theta_msa_est and ln beta_msa_est, 
    the correlation and the covariance matrix.
    Considers the full predictive variance: sampling, true-between-case, model fitting, 
    crossing between sampling and model fitting. 
    """
    var_vfc, *_ = sampling_and_fit_covariance_matrix(model_registry, model, site, n_storeys, ida_tag)
    t = model.T_total
    sigma = var_vfc + t
    se_ln_theta = np.sqrt(sigma[0, 0])
    se_ln_beta = np.sqrt(sigma[1, 1])
    rho = sigma[0, 1] / (se_ln_theta * se_ln_beta)
    return se_ln_theta, se_ln_beta, rho, sigma


def fragility_confidence_interval(
        ln_theta, ln_beta, cov, imls, n_samples, alpha):
    """ returns the alpha/2 and 1-alpha/2 limits of the fraglity curve describe by
    ln_theta and ln_beta. n_samples random samples are generated to calculate the interval"""

    samples = np.random.multivariate_normal([ln_theta, ln_beta], cov, size=n_samples)
    fc = stats.lognorm(s=np.exp(samples[:, 1]), scale=np.exp(samples[:, 0])).cdf(imls[:, np.newaxis])
    lower, upper = np.quantile(fc, [alpha/2, 1 - alpha/2], axis=1)
    return lower, upper


def plot_fragilty_comparison(imls, fc_msa_ss_est, fc_ida_fx, fc_msa_true, lower_ci, upper_ci, alpha):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(imls, fc_msa_ss_est, label="MSA-SS est.")
    ax.plot(imls, fc_ida_fx, label="IDA-FX")
    ax.plot(imls, fc_msa_true, label="MSA-SS true")
    ax.plot(imls, lower_ci, ls="--", color="0.5", label=f"{(1-alpha)*100:.0f}% CI")
    ax.plot(imls, upper_ci, ls="--", color="0.5")

    ax.grid(True, ls="-.", color="0.8")
    ax.set_xlabel("IML (g)")
    ax.set_ylabel("P[Collapse]")

    ax.set_ylim(0, 1.0) 

    leg = ax.legend(facecolor="w", framealpha=1, loc="upper left")
    frame = leg.get_frame()
    frame.set_edgecolor("k")
    ax.set_facecolor('white')
    return fig, ax


def fragility_comparison(
        model_registry, model, site, n_storeys, alpha, iml_params=[0.01, 3.0, 1000],
        plot=False, fig_path: Path=None, plot_show=False, dpi=300):
    """ returns the fragility curves and the CI of the estimated msa fragility
    as well as the fragility parameters and the variance information. 
    If plot is true a plot of the fragility comparison is produced and when a 
    path is provided it is saved without showing it.
    
    iml_params: [start iml, end iml / theta_msa_est, n_points]
    """
    msa_tag, ida_tag = model.contrast.terms
    out = {}

    out["theta_msa_true"], out["beta_msa_true"] = fragility_parameters(model_registry, model, msa_tag).loc[(site, n_storeys)]
    out["theta_ida"], out["beta_ida"] = fragility_parameters(model_registry, model, ida_tag).loc[(site, n_storeys)]
    out["theta_msa_est"], out["beta_msa_est"] = estimate_of_msa_from_ida(model_registry, model, site, n_storeys)
    out["se_ln_theta_msa_est"], out["se_ln_beta_msa_est"], out["rho_msa_est"], out["cov_msa_est"] = msa_estimate_std_errors(model_registry, model, site, n_storeys, ida_tag)

    imls = np.linspace(iml_params[0], iml_params[1]*out["theta_msa_est"], iml_params[2])
    out["imls"] = imls

    fc_msa_est = stats.lognorm(s=out["beta_msa_est"], scale=out["theta_msa_est"]).cdf(imls)
    fc_ida = stats.lognorm(s=out["beta_ida"], scale=out["theta_ida"]).cdf(imls)
    fc_msa_true = stats.lognorm(s=out["beta_msa_true"], scale=out["theta_msa_true"]).cdf(imls)

    out["lower_ci"], out["upper_ci"] = fragility_confidence_interval(
        np.log(out["theta_msa_est"]), np.log(out["beta_msa_est"]), out["cov_msa_est"], 
        imls, 2000, alpha)

    if plot:
        fig, ax = plot_fragilty_comparison(imls, fc_msa_est, fc_ida, fc_msa_true, out["lower_ci"], out["upper_ci"], alpha)
        fig_name = f"site_{site}_{n_storeys}s_fc_comparison"

        if plot and fig_path:
            plt.savefig(fig_path / fig_name, dpi=dpi, facecolor="w")
        if not plot_show:
            plt.close(fig)
        return out, fig, ax
    return out, None, None

############      VARIANCE DECOMPOSITION     ###############
VAR_CONTRIBUTION_TAGS = [r"$V^2_{\theta}$", r"$V^2_{\beta}$", r"$V_{\theta\beta}$",
                         r"$T^2_{\theta}$", r"$T^2_{\beta}$", r"$T_{\theta\beta}$",
                         r"$F^2_{\theta}$", r"$F^2_{\beta}$", r"$F_{\theta\beta}$",
                         r"$C^2_{\theta}$", r"$C^2_{\beta}$", r"$C_{\theta\beta}$",]

VAR_CONTRIBUTION_TAG_MAP = {
    ("sampling", "theta"): r"$V^2_{\theta}$", 
    ("sampling", "beta"): r"$V^2_{\beta}$", 
    ("sampling", "theta_beta"): r"$V_{\theta\beta}$", 
    ("between", "theta"): r"$T^2_{\theta}$", 
    ("between", "beta"): r"$T^2_{\beta}$", 
    ("between", "theta_beta"): r"$T_{\theta\beta}$",
    ("fit", "theta"): r"$F^2_{\theta}$", 
    ("fit", "beta"): r"$F^2_{\beta}$", 
    ("fit", "theta_beta"): r"$F_{\theta\beta}$",
    ("cross", "theta"): r"$C^2_{\theta}$", 
    ("cross", "beta"): r"$C^2_{\beta}$", 
    ("cross", "theta_beta"): r"$C_{\theta\beta}$",
    "total": r"\Sigma"}


def get_A_matrix(model, site, n_storeys):
    standardised_covariates, covariate_names, _ = fm.build_X_joint(model.spec, model.contrast)
    n = model.contrast.n
    i = model.contrast.theta.y_hat.index.get_loc((site, n_storeys))
    # rows of A (theta, beta indicators 0.0 if not a covariate of the quantity.). columns of A (theta coefficients then beta coefficients)
    A = np.vstack([standardised_covariates[i], standardised_covariates[n + i]])
    return A, covariate_names


def fit_covariance_matrix(model, site, n_storeys):
    A, _ = get_A_matrix(model, site, n_storeys)
    V_gamma = model.vcov_beta_boot
    fit_covariance = A @ V_gamma @ A.T
    return fit_covariance


def model_fit_estimates(model, site, n_storeys):
    """ returns the predicted ss/ida correction factor as a log ratio for the estimated ida theta and beta.
    Gets the covariates etc from the model for the given site and n_storeys"""
    A, _ = get_A_matrix(model, site, n_storeys)       # A size = (2, p) where p = number of covariates
    coeffs = model.beta  # regression coefficients for the model. size (1, p) where r = number of replicates
    fit = coeffs @ A.T        # size (r, 2): [theta_fit, beta_fit]
    return fit


def model_fit_replicates(model, site, n_storeys):
    """ returns the predicted ss/ida correction factor as a log ratio for all the replicate fits of the model.
    Gets the covariates etc from the model for the given site and n_storeys"""
    A, covariate_names = get_A_matrix(model, site, n_storeys)       # A size = (2, p) where p = number of covariates
    col_names = [f"beta_{n[0]}_{n[1]}" for n in covariate_names]
    coeff_replicates = model.replicates[col_names].to_numpy()  # regression coefficients for each replicate. size (r, p) where r = number of replicates
    fit = coeff_replicates @ A.T        # size (r, 2): [theta_fit, beta_fit]
    return fit


def sampling_and_fit_covariance_matrix(model_registry, model, site, n_storeys, ida_tag):
    # fit values for each replicate
    fit = model_fit_replicates(model, site, n_storeys)
    
    # replicate estimates of theta and beta
    r = len(model.replicates.to_numpy())
    theta_replicates = model_registry.theta_data.to_rows(ida_tag, model.corrected["theta"])[0].loc[(site, n_storeys)].to_numpy()[:r]  # size (1, r)
    beta_replicates = model_registry.beta_data.to_rows(ida_tag, model.corrected["beta"])[0].loc[(site, n_storeys)].to_numpy()[:r]     # size (1, r)
    ida = np.column_stack([theta_replicates, beta_replicates])       # size (r, 2)
    
    # z(r) -> the estimated theta_ss and beta_ss for each replicate
    z = ida + fit

    # covariances
    total = np.cov(z, rowvar=False)
    V = np.cov(ida, rowvar=False)
    F = np.cov(fit, rowvar=False)
    cross = total - V -F
    return total, V, F, cross


def get_fosm_mafe_slopes(theta, beta, delta, hc_ims, hc_lambda):
    """ returns the slopes d_ln_lambda / d_ln_theta and d_ln_lambda / d_ln_beta.
    
    The slopes are in terms of d_ln_lambda because the function for ln_lambda is
    smoother than for lambda and the fosm approximation (which is linear) is more
    applicable. The slopes are in terms of d_ln_theta/beta because the variances from the
    regression models are in terms of ln_theta/beta and the slopes need to be consistent
    with the variances for the FOSM variance to be meaningful.
    """
    ln_th = np.log(theta)
    ln_bt = np.log(beta)
    d_ln_theta = delta 
    d_ln_beta = delta

    # we do this in log space because the linear approximation of FOSM is closer in log space
    ln_mafc_p_theta = np.log(risk.numerical_mafe_pchip(
        np.exp(ln_th + d_ln_theta), hc_ims, hc_lambda, np.exp(ln_bt)))
    ln_mafc_m_theta = np.log(risk.numerical_mafe_pchip(
        np.exp(ln_th - d_ln_theta), hc_ims, hc_lambda, np.exp(ln_bt)))

    ln_theta_slope = (ln_mafc_p_theta - ln_mafc_m_theta) / (2 * d_ln_theta)

    ln_mafc_p_beta = np.log(risk.numerical_mafe_pchip(
        np.exp(ln_th), hc_ims, hc_lambda, np.exp(ln_bt + d_ln_beta)))
    ln_mafc_m_beta = np.log(risk.numerical_mafe_pchip(
        np.exp(ln_th), hc_ims, hc_lambda, np.exp(ln_bt - d_ln_beta)))

    ln_beta_slope = (ln_mafc_p_beta - ln_mafc_m_beta) / (2 * d_ln_beta)

    return ln_theta_slope, ln_beta_slope
    

def fosm_variance(var_theta, var_beta, cov_theta_beta, d_ln_lam_d_ln_theta, d_ln_lam_d_ln_beta):
    # following Ang and Tang (2007) eq 4.51 pg 186
    var_lam = ((d_ln_lam_d_ln_theta ** 2) * var_theta +                         # theta
               (d_ln_lam_d_ln_beta ** 2) * var_beta +                           # beta    
               2 * d_ln_lam_d_ln_theta * d_ln_lam_d_ln_beta * cov_theta_beta)   # theta beta covariance
    return var_lam


def decompose_2x2_covariance_matrix(covariance_matix):
    """ assumes covariance matrix is 2x2 with theta in row/col 0 and beta in row/col 1 """
    theta = covariance_matix[0, 0]
    beta = covariance_matix[1, 1]
    cov = covariance_matix[0, 1]
    return theta, beta, cov


def variance_log_fragility_parameters(joint, model, site, n_storeys):
    """ returns the variance of ln theta, ln beta and their covariance for
    each of the following sources of variance: 
    (1) sampling error (V),
    (2) true between case variance (T),
    (3) fitting variance of regression model,
    (4) cross correlation between sampling error and fitting variance (C)    
    """

    # Get all the required variance components from the model and the ida results
    # ------ (4) Covariance of sampling error and fit covariance ------
    _, V, F, error_fit_cross = sampling_and_fit_covariance_matrix(
        joint, model, site, n_storeys, "c_avg")
    cross = {}
    cross["theta"], cross["beta"], cross["theta_beta"] = decompose_2x2_covariance_matrix(error_fit_cross)

    # ----- (1) Sampling variance components from the IDA results ------
    # variance of **ln(theta)** and **ln(beta)** from the IDA results
    sampling = {}
    sampling["theta"], sampling["beta"], sampling["theta_beta"] = decompose_2x2_covariance_matrix(V)

    # ----- (2) True variance components from the metaregression model ------
    # variance of **ln(theta_ss/theta_fx)** and **ln(beta_ss/beta_fx)**
    between = {}
    between["theta"], between["beta"], between["theta_beta"] = decompose_2x2_covariance_matrix(model.T_total)    

    # ----- (3) Fitting variance components from the metaregression model ------
    fit = {}
    fit["theta"], fit["beta"], fit["theta_beta"] = decompose_2x2_covariance_matrix(F)

    # Total Variance
    Sigma = model.T_total + V + F + error_fit_cross  
    total = {}
    total["theta"], total["beta"], total["theta_beta"] = decompose_2x2_covariance_matrix(Sigma)

    components = {
        "sampling": sampling,
        "between": between,
        "fit": fit,
        "cross": cross,
        "total": total}

    df = pd.DataFrame.from_dict(components, "index")

    return df


def variance_decomposition_ln_mafe(
        theta_slope, beta_slope, ln_fragility_parameter_variances):
    """ returns a dictionary of the different components that make up the total variance
    of the ln MAFE"""

    var_sources = {}
    var_totals = {}
    for source, components in ln_fragility_parameter_variances.iterrows():
        
        for var_type, val in components.items():
            out = var_totals if source == "total" else var_sources
            tag = (source, var_type)
            if var_type == "theta":
                out[tag] = (fosm_variance(val, 0, 0, theta_slope, beta_slope))
            elif var_type == "beta":
                out[tag] = (fosm_variance(0, val, 0, theta_slope, beta_slope))
            elif var_type == "theta_beta":
                out[tag] = (fosm_variance(0, 0, val, theta_slope, beta_slope))

    # just making sure everything adds up
    var_tot1 = np.array(list(var_sources.values())).sum()
    var_tot2 = np.array(list(var_totals.values())).sum()  # total variance
    assert np.isclose(var_tot1, var_tot2), f"var_tot1 = {var_tot1}, var_tot2 = {var_tot2}"
    
    return var_sources, var_totals, var_tot2  


def convert_ln_mafe_variance(mafe, var_ln_mafe):
    """ converts the total variance on the ln_mafe to the variance of mafe.
    This function assumes that mafe is lognormally distributed."""
    return mafe ** 2 * np.exp(var_ln_mafe) * (np.exp(var_ln_mafe) - 1)


def mafe_variance_decomposition(
        theta, beta, hc_ims, hc_lambdas, model_registry, model, site, n_storeys):
    """ returns the variance decomposition and intermediate results for single
    site/structure combination in the dataset"""
    delta = 0.01    # a small change (1%) in theta and beta for central difference

    theta_slope, beta_slope = get_fosm_mafe_slopes(theta, beta, delta, hc_ims, hc_lambdas)
    var_fc_df = variance_log_fragility_parameters(model_registry, model, site, n_storeys)
    var_source_decomp, var_total_decomp, var_tot = variance_decomposition_ln_mafe(theta_slope, beta_slope, var_fc_df)
    mafe = risk.numerical_mafe_pchip(theta, hc_ims, hc_lambdas, beta)
    out = {
        "fragility_variance": var_fc_df,
        "var_source_decomp": var_source_decomp,
        "total_var_decomp": var_total_decomp,
        "var_tot_ln_mafe": var_tot,
        "var_source_decomp_normalised": {k:v/var_tot for k, v in var_source_decomp.items()},
        "mafe": mafe,
        "var_tot_mafe": convert_ln_mafe_variance(mafe, var_tot)
    }

    return out


def plot_variance_contributions(
    contributions,
    tags,
    ax=None,
    pos_color="#4C72B0",
    neg_color="#DD8452",
    bar_height=0.55,
    stripe_color="0.9",
    grid_color="0.6",
    tag_gutter=0.18,
    xlabel="Contribution to Var[ln MAFC] (-)",
    fp=None,
    plot_show=False
):
    """Horizontal bar chart of the signed contribution of each variance component.

    One bar per component, drawn top-to-bottom in the order given. Positive
    contributions (variances, and covariances that add to the total) are drawn
    in one colour; negative contributions (covariances that reduce the total)
    in another. Alternating white/grey bands sit behind the bars so the eye can
    follow each row across, and each component's tag is written inside the
    axes frame, centred on the mid-height of its band.

    Parameters
    ----------
    contributions : array-like, shape (n,)
        Signed contribution of each component, e.g. its share of the total
        variance (these sum to 1, and individual entries can be negative).
    tags : sequence of str, length n
        Label for each component, in the same order as `contributions`.
        Mathtext labels need a `$` at BOTH ends, e.g. r"$V^2_{\\theta}$".
    ax : matplotlib Axes, optional
        Axes to draw on. A new figure is created if not given.
    pos_color, neg_color : colour
        Fill colours for positive and negative bars.
    bar_height : float
        Bar thickness in row units. Each band is 1.0 high, so any value below
        1.0 leaves the band visible above and below the bar.
    stripe_color : colour
        Colour of the grey bands (the other bands are left white).
    grid_color : colour
        Colour of the dash-dot vertical grid lines.
    tag_gutter : float
        Fraction of the axes width reserved on the left for the tags, so the
        bars never run underneath the text.
    xlabel : str
        x-axis label.
    fp : str or Path, optional
        If given, the figure is saved here with an opaque white background.

    Returns
    -------
    fig, ax
    """
    values = np.asarray(contributions, dtype=float)
    n = len(values)
    if len(tags) != n:
        raise ValueError(f"got {n} contributions but {len(tags)} tags")

    if ax is None:
        # Height scales with the number of rows so the bands keep a readable size.
        fig, ax = plt.subplots(figsize=(6, 0.35 * n + 0.8))
    else:
        fig = ax.figure
    ax.set_facecolor("w")

    # Row i sits at y = i. Inverting the y-axis later puts component 0 at the top,
    # so the chart reads in the same order as the list of tags.
    y = np.arange(n)

    # ---- x-limits, with a gutter on the left kept clear for the tags ----
    # The data range always includes 0 so the zero line is on the plot.
    x_lo = min(values.min(), 0.0)
    x_hi = max(values.max(), 0.0)
    span = x_hi - x_lo
    pad = 0.05 * span  # breathing room beyond the longest bars
    # Choose the left limit so the gutter is exactly `tag_gutter` of the total
    # width: if the data (plus padding) occupy width W, the full axis width is
    # W / (1 - tag_gutter), and the gutter is the remaining tag_gutter share.
    data_w = span + 2 * pad
    gutter_w = tag_gutter / (1.0 - tag_gutter) * data_w
    x_left = x_lo - pad - gutter_w
    x_right = x_hi + pad
    ax.set_xlim(x_left, x_right)

    # ---- alternating background bands ----
    # Each band spans a full row (y-0.5 to y+0.5), so the bands are wider than
    # the bars (bar_height < 1). Every other band is grey; the rest show the
    # white axes background. zorder=0 keeps them behind everything else.
    for yi in y:
        if yi % 2 == 0:
            ax.axhspan(yi - 0.5, yi + 0.5, color=stripe_color, lw=0, zorder=0)

    # ---- bars ----
    # The colour is chosen per bar from the sign of its value.
    colors = np.where(values >= 0, pos_color, neg_color)
    ax.barh(y, values, height=bar_height, color=colors, edgecolor="k",
            linewidth=0.8, zorder=2)

    # Vertical reference line at zero. It is drawn on top of the bars so it stays
    # a continuous line through every row.
    ax.axvline(0.0, color="k", lw=1.0, zorder=3)

    # ---- tags inside the frame ----
    # A blended transform takes x in axes fractions (0 = left frame edge) and y
    # in data units, so each tag sits a fixed distance in from the frame and
    # exactly at the mid-height of its band, whatever the x-limits are.
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for yi, tag in zip(y, tags):
        ax.text(0.02, yi, tag, transform=trans, ha="left", va="center", zorder=4)

    # ---- vertical grid lines ----
    # Only x gridlines: horizontal ones would cut through the bands, which already
    # separate the rows. set_axisbelow(True) puts the grid at zorder 0.5, i.e.
    # above the bands (zorder 0) so it shows on the grey rows, but below the
    # bars (zorder 2) so it never crosses them.
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, ls="-.", color=grid_color, lw=0.8)

    # ---- axes tidy-up ----
    ax.set_ylim(n - 0.5, -0.5)  # inverted: first component at the top, no gaps at the ends
    ax.set_yticks([])           # the tags replace the y tick labels
    ax.set_xlabel(xlabel)

    if fp is not None:
        fig.savefig(fp, dpi=300, bbox_inches="tight", facecolor="w")
    if not plot_show:
        plt.close(fig)
    return fig, ax 


def fragility_variance_assessment(
        model_registry, model, site, n_storeys, hc_ims, hc_lambdas, alpha, iml_params=[0.01, 3.0, 1000],
        plot=False, fc_fig_path: Path=None, var_fig_path: Path=None, plot_show=False, dpi=300):

    fc_out, fig_fc, ax_fc = fragility_comparison(
        model_registry, model, site, n_storeys, alpha, iml_params,
        plot, fc_fig_path, plot_show, dpi)

    vdcomp_out = mafe_variance_decomposition(
        fc_out["theta_msa_est"], fc_out["beta_msa_est"], hc_ims, hc_lambdas, model_registry, model, site, n_storeys)

    mafc_ida = risk.numerical_mafe_pchip(fc_out["theta_ida"], hc_ims, hc_lambdas, fc_out["beta_ida"])
    mafc_msa_true = risk.numerical_mafe_pchip(fc_out["theta_msa_true"], hc_ims, hc_lambdas, fc_out["beta_msa_true"])

    vdcomp_out["mafc_ida"] = mafc_ida
    vdcomp_out["mafc_msa_true"] = mafc_msa_true

    contributions = list(vdcomp_out["var_source_decomp_normalised"].values())
    tags = [VAR_CONTRIBUTION_TAG_MAP[k] for k in vdcomp_out["var_source_decomp_normalised"].keys()]
    fig_name = f"site_{site}_{n_storeys}s_var_decomp.png"
    fig_var, ax_var = plot_variance_contributions(contributions, tags, fp=var_fig_path/fig_name, plot_show=plot_show, xlabel=r"Contribution to Var[$\lambda_C$]")

    return fc_out, vdcomp_out, fig_fc, ax_fc, fig_var, ax_var

############      ASSESSMENT PLOTS ACROSS ALL SITE/STRUCTURE ROWS     ###############
#
# The functions below take the per-row results of `fragility_variance_assessment`
# (one entry per (site, n_storeys) pair, 120 rows in the full study) and turn them
# into summary plots that show, across every row at once:
#   - how each of the 12 variance sources contributes to Var[ln MAFC],
#   - how far the IDA-FX and the corrected (MSA-SS estimated) MAFC are from the
#     MSA-SS "true" MAFC, and whether the correction moves the estimate closer,
#   - whether the predicted uncertainty of the corrected MAFC is honest
#     (standardised errors compared with N(0, 1)).
#
# Every row-wise plot uses the same x-axis: all 3-storey rows (site order) followed
# by all 5-storey rows (site order), separated by a vertical line. Keeping one fixed
# order means a bar at a given position is the same structure in every figure.

# Order of the four variance sources and three terms, used for loops and file names.
VAR_SOURCES = ["sampling", "between", "fit", "cross"]
VAR_TERMS = ["theta", "beta", "theta_beta"]

# Short letter for each source, used in the stacked-share legend.
VAR_SOURCE_LETTER = {"sampling": "V", "between": "T", "fit": "F", "cross": "C"}

# Labels for the three FOSM terms of the TOTAL covariance matrix Sigma.
TOTAL_TAG_MAP = {
    "theta": r"$\Sigma^2_{\theta}$",
    "beta": r"$\Sigma^2_{\beta}$",
    "theta_beta": r"$\Sigma_{\theta\beta}$",
}

# Default duration covariate used to colour the improvement plot: the log ratio of
# the geometric-mean RSD595 of the site-specific records to that of the fixed set.
DURATION_COVARIATE = "d_ln_ds595_mean_unif"
DURATION_COVARIATE_LABEL = r"$\ln(\mu_{RSD595}^{ss}/\mu_{RSD595}^{fx})$"

# Matplotlib's default cycle colours, matching the fragility-curve comparison plots:
# IDA-FX is always orange, the corrected MSA-SS estimate blue and MSA-SS true green,
# so the methods read the same way in every figure.
IDA_COLOUR = "C1"
EST_COLOUR = "C0"
TRUE_COLOUR = "C2"

LN_LAMBDA_LABEL = r"Var[ln $\lambda_C$]"
DELTA_LABEL = r"$\Delta=|\ln(\lambda_{IDA}/\lambda_{MSA})| - |\ln(\lambda_{est}/\lambda_{MSA})|$"


def ln_mafe_sampling_variance(theta, beta, var_ln_theta, var_ln_beta, cov_ln_theta_ln_beta,
                              hc_ims, hc_lambdas, delta=0.01):
    """FOSM sampling variance of ln MAFC for a fragility curve with known parameter
    sampling covariance.

    Used for the MSA-SS "true" curve, whose (ln theta, ln beta) are themselves
    bootstrap estimates. The slopes d ln(lambda)/d ln(theta) and d ln(lambda)/d ln(beta)
    are evaluated at the curve's own parameters, then combined with its parameter
    covariance by the first-order second-moment rule (Ang & Tang 2007, eq. 4.51, p. 186):

        Var[ln lambda] = s_t^2 Var[ln theta] + s_b^2 Var[ln beta] + 2 s_t s_b Cov[ln theta, ln beta]
    """
    s_theta, s_beta = get_fosm_mafe_slopes(theta, beta, delta, hc_ims, hc_lambdas)
    return (fosm_variance(var_ln_theta, 0, 0, s_theta, s_beta)
            + fosm_variance(0, var_ln_beta, 0, s_theta, s_beta)
            + fosm_variance(0, 0, cov_ln_theta_ln_beta, s_theta, s_beta))


def parameter_sampling_covariance(model_registry, model, site, n_storeys, tag):
    """2x2 bootstrap sampling covariance of (ln theta, ln beta) for one arm at one row.

    Uses the same replicates as the regression model: the first r replicates,
    where r is the number of model replicates, with the theta/beta bias
    correction the model was fitted with. This is exactly how the IDA sampling
    term V is built in `sampling_and_fit_covariance_matrix`, so the IDA result
    here equals the "sampling" row of the variance decomposition.

    tag : str
        Arm symbol, e.g. model.contrast.terms -> (msa_tag, ida_tag).
    """
    r = len(model.replicates.to_numpy())
    theta_reps = model_registry.theta_data.to_rows(tag, model.corrected["theta"])[0].loc[(site, n_storeys)].to_numpy()[:r]
    beta_reps = model_registry.beta_data.to_rows(tag, model.corrected["beta"])[0].loc[(site, n_storeys)].to_numpy()[:r]
    return np.cov(np.column_stack([theta_reps, beta_reps]), rowvar=False)


def mafe_confidence_intervals(df, level=0.95):
    """Add confidence intervals, z-values and inside-CI flags for the three MAFCs.

    Needs the columns var_ln_mafe_true and var_ln_mafe_ida (FOSM sampling
    variances, added from the bootstrap replicates) and var_tot (the full
    predictive variance of the corrected estimate from the decomposition).

    Assuming ln lambda is normal, the CI of each method m is

        lambda_m * exp(+/- z_{(1+level)/2} * sqrt(Var[ln lambda_m]))

    which is symmetric on a log axis. Added columns:
        var_ln_mafe_est           = var_tot
        ci_lo_{m}, ci_hi_{m}      for m in true, ida, est
        z_ida, z_est              ln(lambda / lambda_true) / sqrt(Var[ln lambda_true]):
                                  distance from the MSA-SS value in units of
                                  MSA-SS's own uncertainty
        ida_in_ci, est_in_ci      inside the MSA-SS CI (identical to |z| <= z_crit)
        z_ida_combined, z_est_combined
                                  ln(lambda / lambda_true) / sqrt(Var[ln lambda] + Var[ln lambda_true]):
                                  the same distance in units of BOTH methods'
                                  uncertainty, as used in `plot_z_calibration`
    """
    df = df.copy()
    z_crit = stats.norm.ppf(0.5 + level / 2)
    df["var_ln_mafe_est"] = df["var_tot"]
    for m in ("true", "ida", "est"):
        sd = np.sqrt(df[f"var_ln_mafe_{m}"])
        df[f"ci_lo_{m}"] = df[f"mafe_{m}"] * np.exp(-z_crit * sd)
        df[f"ci_hi_{m}"] = df[f"mafe_{m}"] * np.exp(z_crit * sd)
    sd_true = np.sqrt(df["var_ln_mafe_true"])
    for m in ("ida", "est"):
        df[f"z_{m}"] = df[f"ln_ratio_{m}"] / sd_true
        df[f"{m}_in_ci"] = (df[f"mafe_{m}"] >= df["ci_lo_true"]) & (df[f"mafe_{m}"] <= df["ci_hi_true"])
        df[f"z_{m}_combined"] = df[f"ln_ratio_{m}"] / np.sqrt(df[f"var_ln_mafe_{m}"] + df["var_ln_mafe_true"])
    return df


def mafe_ci_summary(df):
    """Count, per storey group and overall, the rows whose IDA and corrected-estimate
    MAFC lie inside the MSA-SS confidence interval (columns from `mafe_confidence_intervals`)."""
    by_storey = df.groupby(level="n_storeys")[["ida_in_ci", "est_in_ci"]].agg(["sum", "count"])
    out = pd.DataFrame({
        "IDA-FX inside": by_storey[("ida_in_ci", "sum")],
        "MSA-SS est. inside": by_storey[("est_in_ci", "sum")],
        "rows": by_storey[("ida_in_ci", "count")],
    })
    out.loc["all"] = [df["ida_in_ci"].sum(), df["est_in_ci"].sum(), len(df)]
    out.index.name = "n_storeys"
    return out.astype(int)


def assessment_frame(results, data, duration_covariate=DURATION_COVARIATE):
    """Collect the per-row assessment results into one table for plotting.

    Parameters
    ----------
    results : dict
        {(site, n_storeys): output tuple of `fragility_variance_assessment`}.
        Only the first two entries of each tuple are used: the fragility
        comparison dict and the variance decomposition dict.
    data : pandas.DataFrame
        The fragility regression dataset (one row per site/structure). Supplies
        the design group and the duration covariate.
    duration_covariate : str
        Column of `data` copied into the table for colouring the improvement plot.

    Returns
    -------
    pandas.DataFrame indexed by (site, n_storeys), ordered by (n_storeys, site), with:
        var_{source}_{term}   contribution of each of the 12 sources to Var[ln MAFC]
        pct_{source}_{term}   the same as a percentage of the total
        tot_{term}            FOSM terms of the total covariance (sum to var_tot)
        var_tot               Var[ln MAFC] of the corrected estimate
        mafe_true, mafe_ida, mafe_est
                              MAFC from MSA-SS (true), IDA-FX and the corrected estimate
        ln_ratio_ida, ln_ratio_est
                              ln(lambda / lambda_true) for IDA and for the estimate
        improvement           Delta = |ln_ratio_ida| - |ln_ratio_est|
                              (+ = estimate closer to MSA-SS than IDA is)
        theta_msa_true, beta_msa_true, theta_ida, beta_ida
                              fragility parameters, for the FOSM slopes of the
                              MSA-SS and IDA MAFC (see `ln_mafe_sampling_variance`)
        design_group_id, n_sites_in_group, <duration_covariate>

    The sampling variances of ln lambda for MSA-SS and IDA are not part of this
    table; they are added afterwards (columns var_ln_mafe_true, var_ln_mafe_ida)
    from the bootstrap replicates, and are needed by `plot_z_calibration` and
    `mafe_confidence_intervals`.
    """
    rows = []
    for (site, n_storeys), out in results.items():
        fc_out, vd_out = out[0], out[1]
        row = {"site": site, "n_storeys": n_storeys}

        # The 12 source contributions (variances in ln-MAFC units) and the three
        # FOSM terms of the total covariance. Their sums are both var_tot.
        for (source, term), val in vd_out["var_source_decomp"].items():
            row[f"var_{source}_{term}"] = val
        for (_, term), val in vd_out["total_var_decomp"].items():
            row[f"tot_{term}"] = val
        row["var_tot"] = vd_out["var_tot_ln_mafe"]

        # MAFC of the three curves. `mafe` in the decomposition dict is the
        # corrected estimate, because the decomposition is evaluated at the
        # estimated (theta, beta).
        row["mafe_true"] = vd_out["mafc_msa_true"]
        row["mafe_ida"] = vd_out["mafc_ida"]
        row["mafe_est"] = vd_out["mafe"]

        # MSA-SS and IDA parameters, needed for the FOSM slopes of their MAFC.
        row["theta_msa_true"] = fc_out["theta_msa_true"]
        row["beta_msa_true"] = fc_out["beta_msa_true"]
        row["theta_ida"] = fc_out["theta_ida"]
        row["beta_ida"] = fc_out["beta_ida"]
        rows.append(row)

    df = pd.DataFrame(rows).set_index(["site", "n_storeys"])

    # Percentage share of each source in the total.
    for source in VAR_SOURCES:
        for term in VAR_TERMS:
            df[f"pct_{source}_{term}"] = 100 * df[f"var_{source}_{term}"] / df["var_tot"]

    # Log errors relative to MSA-SS. The log ratio is symmetric (a factor of 2
    # high and a factor of 2 low are equal and opposite), unlike a % difference.
    df["ln_ratio_ida"] = np.log(df["mafe_ida"] / df["mafe_true"])
    df["ln_ratio_est"] = np.log(df["mafe_est"] / df["mafe_true"])
    df["improvement"] = df["ln_ratio_ida"].abs() - df["ln_ratio_est"].abs()

    # Columns brought in from the regression dataset.
    extra_cols = ["design_group_id", "n_sites_in_group", duration_covariate]
    df = df.join(data.set_index(["site", "n_storeys"])[extra_cols])

    # Fixed plotting order: 3-storey rows first, then 5-storey, each in site order.
    return df.sort_index(level=["n_storeys", "site"])


def _mean_sd_text(values):
    """'mean = ..., SD = ...' with 3 significant figures; SD uses ddof=1 (sample SD)."""
    values = np.asarray(values, dtype=float)
    return f"mean = {values.mean():.3g},  SD = {values.std(ddof=1):.3g}"


def _stats_box(ax, lines, loc="upper right"):
    """Write summary lines in a boxed annotation in a top corner of the axes.

    The y-limit is first raised so the box sits in empty headroom instead of on
    top of the tallest bars: the extra height grows with the number of lines.
    """
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (0.06 * len(lines) + 0.05) * (hi - lo))
    x, ha = {"upper right": (0.98, "right"), "upper left": (0.02, "left")}[loc]
    ax.text(x, 0.96, "\n".join(lines), transform=ax.transAxes, ha=ha, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="w", ec="0.4"), zorder=5)


def _site_axis_frame(ax, df, ylabel):
    """Shared x-axis for the row-wise bar charts.

    Bars sit at x = 0..n-1 in the order of `df` (3-storey then 5-storey). Site
    numbers are ticked every 5 sites; a vertical line separates the two storey
    groups, and each group is labelled under its half of the axis.
    """
    sites = df.index.get_level_values("site").to_numpy()
    n_storeys = df.index.get_level_values("n_storeys").to_numpy()
    x = np.arange(len(df))

    # Boundary between the storey groups, half-way between the last row of one
    # group and the first of the next.
    for n in np.unique(n_storeys)[:-1]:
        ax.axvline(np.where(n_storeys == n)[0][-1] + 0.5, color="k", lw=0.8)

    # Group labels: x in data units (centre of the group), y in axes units so the
    # text sits a fixed distance below the frame whatever the y-limits are.
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for n in np.unique(n_storeys):
        idx = np.where(n_storeys == n)[0]
        ax.text(idx.mean(), -0.09, f"Site ({n}-storey)", transform=trans,
                ha="center", va="top", fontsize=10)

    ticks = x[sites % 5 == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels(sites[sites % 5 == 0], fontsize=8)
    ax.set_xlim(-1, len(df))
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="k", lw=0.8)
    ax.yaxis.grid(True, ls="-.", color="0.8")
    ax.set_axisbelow(True)
    ax.set_facecolor("w")


def _add_ef_ratio_axis(ax):
    """Right-hand y-axis showing e^Delta = EF_IDA / EF_est.

    The error factor EF = exp(|ln(lambda/lambda_true)|) is how many times too high
    or too low a MAFC is. Delta is a difference of log errors, so e^Delta is the
    ratio of the two error factors: e^Delta = 1.65 means IDA's error factor is
    1.65 times the estimate's. `secondary_yaxis` maps the left axis through exp
    (and back through log), so each right-hand tick sits at Delta = ln(tick).
    Call this after the y-limits are final.

    The inverse is a log clipped to the smallest positive float. e^Delta is always
    > 0, so the clip never touches a real value; it only catches the 0 that
    matplotlib passes in internally when placing the axis labels (the hidden
    bottom spine of the secondary axes), which a plain np.log reports as a
    divide-by-zero warning.
    """
    safe_log = lambda v: np.log(np.clip(v, np.finfo(float).tiny, None))
    sec = ax.secondary_yaxis("right", functions=(np.exp, safe_log))
    lo, hi = ax.get_ylim()
    ticks = [t for t in [0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0] if lo <= np.log(t) <= hi]
    sec.set_yticks(ticks)
    sec.set_yticklabels([f"{t:g}" for t in ticks])
    sec.set_ylabel(r"$e^{\Delta}=EF_{IDA}\,/\,EF_{est}$")
    return sec


def _improvement_stats_lines(improvement):
    """Summary lines for the improvement plots.

    exp(mean Delta) is the geometric mean of e^Delta, the typical ratio of IDA's
    error factor to the estimate's. The geometric mean is the right average for a
    ratio: it treats 'twice as large' and 'half as large' as equal and opposite.
    """
    imp = np.asarray(improvement, dtype=float)
    return [r"$\Delta$: " + _mean_sd_text(imp),
            r"geometric mean of $e^{\Delta}$ = " + f"{np.exp(imp.mean()):.3f}",
            f"improved: {np.sum(imp > 0)}/{len(imp)} rows"]


def _finish(fig, fp, dpi, plot_show):
    """Save with an opaque white background (if a path is given) and close unless shown."""
    if fp is not None:
        fig.savefig(fp, dpi=dpi, bbox_inches="tight", facecolor="w")
    if not plot_show:
        plt.close(fig)


def plot_site_bars(df, column, ylabel, title, fp=None, pos_color=EST_COLOUR, neg_color=IDA_COLOUR,
                   dpi=300, plot_show=False):
    """One bar per site/structure row for a single column of `df`.

    Used for the % contribution, the absolute contribution, and the total-variance
    terms. Positive bars use `pos_color` and negative bars `neg_color`, so
    covariance terms that reduce the total stand out. The red dashed line is the
    mean across rows, and the box in the top-right corner gives the mean and SD.
    """
    values = df[column].to_numpy(dtype=float)
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.bar(x, values, width=0.8, color=np.where(values >= 0, pos_color, neg_color), edgecolor="none")
    ax.axhline(values.mean(), color="C3", ls="--", lw=1)
    _site_axis_frame(ax, df, ylabel)
    ax.set_title(title)
    _stats_box(ax, [_mean_sd_text(values)])
    _finish(fig, fp, dpi, plot_show)
    return fig, ax


def plot_variance_component_sets(df, fig_dir, dpi=300, plot_show=False):
    """Save the 12 % plots, the 12 absolute plots and the 3 total-variance plots.

    File names:
        var_pct_{source}_{term}.png   share of Var[ln MAFC] (%)
        var_abs_{source}_{term}.png   contribution in ln-MAFC variance units
        var_total_{term}.png          FOSM term of the total covariance
    """
    fig_dir = Path(fig_dir)
    for source in VAR_SOURCES:
        for term in VAR_TERMS:
            tag = VAR_CONTRIBUTION_TAG_MAP[(source, term)]
            plot_site_bars(df, f"pct_{source}_{term}", f"Contribution to {LN_LAMBDA_LABEL} (%)",
                           f"{tag} — share of total variance of ln MAFC",
                           fp=fig_dir / f"var_pct_{source}_{term}.png", dpi=dpi, plot_show=plot_show)
            plot_site_bars(df, f"var_{source}_{term}", f"{LN_LAMBDA_LABEL} contribution (-)",
                           f"{tag} — contribution to Var[ln MAFC]",
                           fp=fig_dir / f"var_abs_{source}_{term}.png", dpi=dpi, plot_show=plot_show)
    for term in VAR_TERMS:
        plot_site_bars(df, f"tot_{term}", f"{LN_LAMBDA_LABEL} contribution (-)",
                       f"Total-variance component: {TOTAL_TAG_MAP[term]} term",
                       fp=fig_dir / f"var_total_{term}.png", dpi=dpi, plot_show=plot_show)


def plot_ln_mafe_ratio(df, fp=None, dpi=300, plot_show=False):
    """Paired bars of ln(lambda / lambda_true) per row: IDA-FX (orange) and the
    corrected estimate (blue). 0 is a perfect match; equal distances above and
    below 0 are errors of the same size. The box gives each set's mean and SD.
    """
    x = np.arange(len(df))
    ln_ida = df["ln_ratio_ida"].to_numpy()
    ln_est = df["ln_ratio_est"].to_numpy()
    w = 0.42  # each bar takes just under half the slot, so a pair fills it

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(x - w / 2, ln_ida, width=w, color=IDA_COLOUR, label=r"IDA-FX: $\ln(\lambda_{IDA}/\lambda_{MSA})$")
    ax.bar(x + w / 2, ln_est, width=w, color=EST_COLOUR, label=r"MSA est.: $\ln(\lambda_{est}/\lambda_{MSA})$")
    _site_axis_frame(ax, df, r"$\ln(\lambda/\lambda_{MSA})$")
    ax.set_title("MAFC relative to MSA-SS (true), log scale")
    _stats_box(ax, ["IDA-FX:    " + _mean_sd_text(ln_ida), "MSA est.:  " + _mean_sd_text(ln_est)], "upper left")
    ax.legend(loc="upper right", fontsize=9, facecolor="w", framealpha=1, edgecolor="k")
    _finish(fig, fp, dpi, plot_show)
    return fig, ax


def plot_mafe_improvement(df, fp=None, colour_by=None, colour_label=None, cmap="RdBu_r",
                          dpi=300, plot_show=False):
    """Bar per row of Delta = |ln(lambda_IDA/lambda_true)| - |ln(lambda_est/lambda_true)|.

    Positive means the corrected estimate is closer to MSA-SS than IDA is. The
    right-hand axis shows e^Delta = EF_IDA / EF_est (see `_add_ef_ratio_axis`).

    colour_by : str, optional
        None  -> blue for improvement, orange for worse.
        column name -> bars coloured by that column with a diverging map (white at
        0) and a colour bar. The colour limits are symmetric, +/- max|value|, so a
        given colour strength means the same size of value on either side of 0.
    """
    x = np.arange(len(df))
    imp = df["improvement"].to_numpy()

    fig, ax = plt.subplots(figsize=(14 if colour_by else 13, 4))
    if colour_by is None:
        ax.bar(x, imp, width=0.8, color=np.where(imp >= 0, EST_COLOUR, IDA_COLOUR))
        title = "Improvement of MSA est. over IDA-FX   (+ = closer to MSA-SS, − = further away)"
    else:
        cv = df[colour_by].to_numpy(dtype=float)
        vmax = np.max(np.abs(cv))
        norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
        cmap = mpl.colormaps[cmap]
        # Thin black edges keep near-white bars visible against the background.
        ax.bar(x, imp, width=0.8, color=cmap(norm(cv)), edgecolor="k", linewidth=0.3)
        title = "Improvement of MSA est. over IDA-FX, coloured by duration covariate"
    _site_axis_frame(ax, df, DELTA_LABEL)
    ax.set_title(title)
    _stats_box(ax, _improvement_stats_lines(imp), "upper left")
    _add_ef_ratio_axis(ax)
    if colour_by is not None:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, pad=0.07, fraction=0.03)  # pad clears the right-hand axis
        cb.set_label(colour_label or colour_by)
    _finish(fig, fp, dpi, plot_show)
    return fig, ax


def plot_mafe_improvement_by_design_group(df, fp=None, dpi=300, plot_show=False):
    """The improvement Delta re-ordered so each design group is a contiguous block.

    Within each storey group the design groups run largest first, with sites in
    ascending order inside a group; designs used at only one site are gathered
    into a single block on the right. Alternating grey bands mark the groups, the
    group number is written under each band, and a black line marks each group's
    mean Delta. Bars keep the blue (better) / orange (worse) colouring.

    The x-axis is NOT in site order, so this figure does not line up bar-for-bar
    with the other row-wise plots. Design groups never span storey groups, so the
    3-storey and 5-storey blocks are independent.
    """
    group = df["design_group_id"].to_numpy()
    size = df["n_sites_in_group"].to_numpy()
    sites = df.index.get_level_values("site").to_numpy()
    n_storeys = df.index.get_level_values("n_storeys").to_numpy()
    imp_all = df["improvement"].to_numpy()

    # Sort key: storey, singletons last, larger groups first, group id, then site.
    order = np.array(sorted(range(len(df)),
                            key=lambda i: (n_storeys[i], size[i] == 1, -size[i], group[i], sites[i])))
    g_o, n_o, k_o, imp = group[order], n_storeys[order], size[order], imp_all[order]
    x = np.arange(len(order))

    # Contiguous blocks of the re-ordered rows: one per shared design group, and
    # one per storey group for all the single-site designs.
    block_key = [(n_o[i], "single" if k_o[i] == 1 else g_o[i]) for i in range(len(order))]
    blocks, start = [], 0
    for i in range(1, len(order) + 1):
        if i == len(order) or block_key[i] != block_key[start]:
            blocks.append((start, i - 1, block_key[start][1]))
            start = i

    fig, ax = plt.subplots(figsize=(14, 4.4))
    shade = 0
    for a, b, name in blocks:
        if name == "single":
            ax.axvspan(a - 0.5, b + 0.5, color="0.97", lw=0, zorder=0)
        else:
            if shade % 2 == 0:
                ax.axvspan(a - 0.5, b + 0.5, color="0.88", lw=0, zorder=0)
            shade += 1
            gm = imp[a:b + 1].mean()
            ax.plot([a - 0.4, b + 0.4], [gm, gm], color="k", lw=1.4, zorder=3)
    ax.bar(x, imp, width=0.8, color=np.where(imp >= 0, EST_COLOUR, IDA_COLOUR), zorder=2)

    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlim(-1, len(order))
    split = np.sum(n_o == n_o[0]) - 0.5
    ax.axvline(split, color="k", lw=0.8)

    # Group numbers under each band ('group_3s_04' -> '04'), and the storey labels below.
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.set_xticks([])
    for a, b, name in blocks:
        label = "one site each" if name == "single" else str(name).split("_")[-1]
        ax.text((a + b) / 2, -0.02, label, transform=trans, ha="center", va="top", fontsize=7)
    for n in np.unique(n_o):
        idx = np.where(n_o == n)[0]
        ax.text(idx.mean(), -0.09, f"Design group ({n}-storey)", transform=trans,
                ha="center", va="top", fontsize=10)

    ax.set_ylabel(DELTA_LABEL)
    ax.yaxis.grid(True, ls="-.", color="0.8")
    ax.set_axisbelow(True)
    ax.set_facecolor("w")
    ax.set_title("Improvement of MSA est. over IDA-FX, grouped by design group (black line = group mean)")
    _stats_box(ax, _improvement_stats_lines(imp), "upper right")
    _add_ef_ratio_axis(ax)
    _finish(fig, fp, dpi, plot_show)
    return fig, ax


def plot_mafe_parity(df, fp=None, include_est=True, dpi=300, plot_show=False):
    """Predicted MAFC against MSA-SS true MAFC on log-log axes.

    Open orange circles are IDA-FX, filled blue circles the corrected estimate. The
    solid line is perfect agreement; the dotted lines are a factor of 2 either way.

    include_est : bool
        True  -> IDA-FX and the corrected estimate.
        False -> IDA-FX only.
    The axis limits are set from all three MAFCs in either case, so the two
    versions have identical axes and can be compared side by side.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.loglog(df["mafe_true"], df["mafe_ida"], "o", ms=4, color=IDA_COLOUR, mfc="none", label="IDA-FX")
    if include_est:
        ax.loglog(df["mafe_true"], df["mafe_est"], "o", ms=4, color=EST_COLOUR, label="MSA est.")
    allv = df[["mafe_true", "mafe_ida", "mafe_est"]].to_numpy()
    lo, hi = allv.min() / 1.3, allv.max() * 1.3
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    for f in (0.5, 2.0):
        ax.plot([lo, hi], [lo * f, hi * f], "k:", lw=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\lambda_C$ MSA-SS true (1/yr)")
    ax.set_ylabel(r"$\lambda_C$ predicted (1/yr)")
    ax.grid(True, which="both", ls="-.", color="0.85")
    ax.set_facecolor("w")
    ax.legend(facecolor="w", framealpha=1, edgecolor="k")
    ax.set_title("Parity: dotted = factor of 2")
    _finish(fig, fp, dpi, plot_show)
    return fig, ax


def plot_mafe_ratio_vs_duration(df, fp=None, include_est=True,
                                duration_covariate=DURATION_COVARIATE, dpi=300, plot_show=False):
    """MAFC ratio lambda / lambda_true (linear, not logged) against the duration ratio.

    x is mu_RSD595^ss / mu_RSD595^fx, the model's duration covariate un-logged
    (exp of `duration_covariate`), so x = 1 (dashed vertical line) is where the
    site-specific and fixed-set records have the same mean duration. The solid
    horizontal line at 1 is perfect agreement with MSA-SS.

    IDA-FX: open orange squares. Corrected MSA-SS estimate: blue diamonds with black
    edges (only if include_est). Each method has a dashed least-squares straight
    line of ratio on duration ratio, with its slope and correlation r in the box:
    a slope near 0 means the method's error does not depend on duration.

    The axis limits are set from both methods in either case, so the IDA-only and
    full versions have identical axes and can be compared side by side.
    """
    x = np.exp(df[duration_covariate].to_numpy(dtype=float))
    ratios = {"ida": (df["mafe_ida"] / df["mafe_true"]).to_numpy(),
              "est": (df["mafe_est"] / df["mafe_true"]).to_numpy()}
    # (method, legend label, colour, marker, extra kwargs) in drawing order
    series = [("ida", "IDA-FX", IDA_COLOUR, "s", dict(ms=5, mfc="none", mew=1.2, zorder=3))]
    if include_est:
        series.append(("est", "MSA-SS est.", EST_COLOUR, "D", dict(ms=4.5, mec="k", mew=0.6, zorder=4)))

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.axhline(1, color="k", lw=0.9)
    ax.axvline(1, color="0.4", lw=0.8, ls="--")

    xx = np.linspace(x.min(), x.max(), 50)
    lines = ["trend (dashed):"]
    for m, label, colour, marker, kw in series:
        ax.plot(x, ratios[m], ls="none", marker=marker, color=colour, label=label, **kw)
        fit = stats.linregress(x, ratios[m])
        ax.plot(xx, fit.intercept + fit.slope * xx, "--", color=colour, lw=1.3, zorder=2)
        lines.append(f"{label}: slope = {fit.slope:+.3f},  r = {fit.rvalue:+.2f}")

    # Limits from both methods (plus margins), fixed before the stats box adds headroom.
    xpad = 0.05 * (x.max() - x.min())
    ax.set_xlim(x.min() - xpad, x.max() + xpad)
    y_all = np.concatenate(list(ratios.values()))
    ypad = 0.05 * (y_all.max() - y_all.min())
    ax.set_ylim(y_all.min() - ypad, y_all.max() + ypad)

    ax.set_xlabel(r"Duration ratio $\mu_{RSD595}^{ss}/\mu_{RSD595}^{fx}$")
    ax.set_ylabel(r"$\lambda/\lambda_{MSA}$")
    ax.grid(True, ls="-.", color="0.85")
    ax.set_axisbelow(True)
    ax.set_facecolor("w")
    ax.legend(loc="upper right", fontsize=9, facecolor="w", framealpha=1, edgecolor="k")
    _stats_box(ax, lines, "upper left")
    # `_stats_box` adds headroom in proportion to its number of lines. Reset the top
    # to the headroom of the full (three-line) box, so both versions share one y-axis.
    lo, hi = y_all.min() - ypad, y_all.max() + ypad
    ax.set_ylim(lo, hi + (0.06 * 3 + 0.05) * (hi - lo))
    _finish(fig, fp, dpi, plot_show)
    return fig, ax


def plot_mafe_correction_calibration(df, fp=None, dpi=300, plot_show=False):
    """Correction needed against correction applied, both as log ratios to IDA-FX.

        x = ln(lambda_est / lambda_IDA)    correction the model applied to IDA
        y = ln(lambda_MSA / lambda_IDA)    correction IDA needed to reach MSA-SS

    0 on either axis means no correction (dotted lines); the solid line is 1:1,
    perfect correction. The dashed red line is the OLS fit of y on x, with its
    slope, intercept (each with a 95% CI from the t distribution) and R^2 in the box.

    The fit is of needed ON applied because that is the calibration test: for a
    well-calibrated predictor the slope is 1 (each extra unit of correction applied
    matches an extra unit needed) and the intercept is 0. A non-zero intercept with
    slope 1 is a constant offset: the estimate is off by a factor e^intercept at
    every row. Fitting the other way round (applied on needed) gives a slope near
    R^2 even for a perfect predictor, because the needed correction contains
    site-specific scatter and MSA-SS sampling noise that no prediction can follow
    (regression to the mean).

    Both axes share one symmetric range about 0, with equal aspect, so the 1:1
    line runs at 45 degrees.
    """
    x = np.log(df["mafe_est"] / df["mafe_ida"]).to_numpy()
    y = np.log(df["mafe_true"] / df["mafe_ida"]).to_numpy()
    fit = stats.linregress(x, y)
    t_crit = stats.t.ppf(0.975, len(x) - 2)   # two-sided 95%, n - 2 degrees of freedom

    m = max(np.abs(np.concatenate([x, y])).max(), 0.1) * 1.12
    lo, hi = -m, m

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.9, label="1:1 (perfect correction)")
    ax.axhline(0, color="0.5", lw=0.7, ls=":")
    ax.axvline(0, color="0.5", lw=0.7, ls=":")
    ax.plot(x, y, "D", ms=4.5, color=EST_COLOUR, mec="k", mew=0.6, label="site/structure rows", zorder=3)
    xx = np.array([lo, hi])
    ax.plot(xx, fit.intercept + fit.slope * xx, "--", color="C3", lw=1.5, label="OLS fit", zorder=2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\ln(\lambda_{est}/\lambda_{IDA})$  (correction applied)")
    ax.set_ylabel(r"$\ln(\lambda_{MSA}/\lambda_{IDA})$  (correction needed)")
    ax.grid(True, ls="-.", color="0.85")
    ax.set_axisbelow(True)
    ax.set_facecolor("w")
    lines = [
        f"slope = {fit.slope:.3f}  (95% CI {fit.slope - t_crit * fit.stderr:.2f} "
        f"to {fit.slope + t_crit * fit.stderr:.2f})",
        f"intercept = {fit.intercept:.3f}  (95% CI {fit.intercept - t_crit * fit.intercept_stderr:.2f} "
        f"to {fit.intercept + t_crit * fit.intercept_stderr:.2f})",
        f"$R^2$ = {fit.rvalue ** 2:.3f},  n = {len(x)}",
    ]
    _stats_box(ax, lines, "upper left")
    ax.set_ylim(lo, hi)   # undo the stats-box headroom: the axes stay square and symmetric
    ax.legend(loc="lower right", fontsize=8.5, facecolor="w", framealpha=1, edgecolor="k")
    _finish(fig, fp, dpi, plot_show)
    return fig, ax


def plot_z_calibration(df, fp=None, dpi=300, plot_show=False):
    """Histogram of standardised errors of the corrected MAFC and of IDA-FX, against N(0, 1).

        z = ln(lambda / lambda_true) / sqrt(Var[ln lambda] + Var[ln lambda_true])

    for lambda = the corrected estimate (Var = var_tot, its full predictive
    variance) and for lambda = IDA-FX (Var = its FOSM sampling variance). If a
    method's stated uncertainty is honest, its z follows a standard normal: mean 0,
    SD 1, and about 95% of rows inside +/-1.96. The two variances add because the
    IDA (fixed set) and MSA-SS (site-specific) arms use disjoint record sets.

    Filled blue: the estimate, including Var[ln lambda_true]. Dashed grey outline:
    the estimate without Var[ln lambda_true], to show how much the sampling error
    of the reference itself matters. Orange outline: IDA-FX, including
    Var[ln lambda_true]; IDA's variance holds sampling error only, so a spread
    wider than N(0, 1) is the between-site scatter that IDA does not account for.

    The regression was fitted on these same MSA-SS values, which makes the
    estimate and the truth slightly positively correlated; that is not removed
    here, so z for the estimate is slightly optimistic.
    """
    var_true = df["var_ln_mafe_true"].to_numpy()
    z_est_excl = df["ln_ratio_est"].to_numpy() / np.sqrt(df["var_tot"].to_numpy())
    z_est = df["ln_ratio_est"].to_numpy() / np.sqrt(df["var_tot"].to_numpy() + var_true)
    z_ida = df["ln_ratio_ida"].to_numpy() / np.sqrt(df["var_ln_mafe_ida"].to_numpy() + var_true)
    inside = lambda z: 100 * np.mean(np.abs(z) < 1.96)

    # Symmetric bin range wide enough for the widest set, in steps of 0.5.
    lim = max(4.0, np.ceil(np.max(np.abs(np.concatenate([z_est, z_est_excl, z_ida])))))
    bins = np.arange(-lim, lim + 0.01, 0.5)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.hist(z_est, bins=bins, density=True, color=EST_COLOUR, edgecolor="k", lw=0.6,
            label=r"MSA-SS est., incl. Var of $\lambda_{MSA}$")
    ax.hist(z_est_excl, bins=bins, density=True, histtype="step", color="0.35", ls="--", lw=1.2,
            label=r"MSA-SS est., excl. Var of $\lambda_{MSA}$")
    ax.hist(z_ida, bins=bins, density=True, histtype="step", color=IDA_COLOUR, lw=1.8,
            label=r"IDA-FX, incl. Var of $\lambda_{MSA}$")
    zz = np.linspace(-lim, lim, 400)
    ax.plot(zz, stats.norm.pdf(zz), "k-", label="N(0,1)")
    _stats_box(ax, ["est. incl.: " + _mean_sd_text(z_est), f"                inside ±1.96: {inside(z_est):.0f}%",
                    "est. excl.: " + _mean_sd_text(z_est_excl), f"                inside ±1.96: {inside(z_est_excl):.0f}%",
                    "IDA-FX:    " + _mean_sd_text(z_ida), f"                inside ±1.96: {inside(z_ida):.0f}%"],
               "upper left")
    ax.set_xlabel(r"$z=\ln(\lambda/\lambda_{MSA})\,/\,\sqrt{\mathrm{Var}[\ln\lambda]"
                  r"+\mathrm{Var}[\ln\lambda_{MSA}]}$")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=8, facecolor="w", framealpha=1, edgecolor="k")
    ax.grid(True, ls="-.", color="0.85")
    ax.set_axisbelow(True)
    ax.set_facecolor("w")
    _finish(fig, fp, dpi, plot_show)
    return fig, ax


def plot_variance_source_shares(df, fp=None, dpi=300, plot_show=False):
    """Stacked bars per row: share of Var[ln MAFC] from each of the four sources
    (V sampling, T between-case, F fit, C cross), each summed over its theta,
    beta and theta-beta terms.

    Positive shares stack upwards from 0 and negative shares downwards, so the
    bars above 0 minus the bars below 0 total 100% for every row.
    """
    x = np.arange(len(df))
    colours = {"sampling": "C0", "between": "C2", "fit": "C4", "cross": "C3"}
    pos = np.zeros(len(df))
    neg = np.zeros(len(df))

    fig, ax = plt.subplots(figsize=(11, 3.8))
    for source in VAR_SOURCES:
        share = sum(df[f"pct_{source}_{term}"] for term in VAR_TERMS).to_numpy()
        bottom = np.where(share >= 0, pos, neg)
        ax.bar(x, share, bottom=bottom, width=0.85, color=colours[source],
               label=f"{VAR_SOURCE_LETTER[source]} ({source})")
        pos += np.clip(share, 0, None)
        neg += np.clip(share, None, 0)
    _site_axis_frame(ax, df, f"Share of {LN_LAMBDA_LABEL} (%)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.17), fontsize=9, frameon=False)
    _finish(fig, fp, dpi, plot_show)
    return fig, ax


def _five_site_bands(ax, df, color="0.92"):
    """Alternating grey/white vertical bands every 5 sites (0-4 grey, 5-9 white,
    10-14 grey, ...) behind a row-wise plot.

    A band starts at every site number divisible by 10 and runs over the following
    consecutive sites up to the next multiple of 5. The pattern restarts in each
    storey group because the site numbers restart there, so the bands always line
    up with the site tick labels of `_site_axis_frame`. zorder=0 keeps them behind
    the grid lines and the data.
    """
    sites = df.index.get_level_values("site").to_numpy()
    for start in np.where(sites % 10 == 0)[0]:
        end = start
        while end + 1 < len(sites) and sites[end + 1] == sites[end] + 1 and sites[end + 1] % 10 < 5:
            end += 1
        ax.axvspan(start - 0.5, end + 0.5, color=color, lw=0, zorder=0)


def plot_mafe_by_site(df, include_est=True, fp=None, dpi=300, plot_show=False):
    """MAFC of each site/structure row on a log axis, one column of markers per row.

    Markers (colours match the fragility-curve comparison plots):
        IDA-FX          open orange squares
        MSA-SS est.     blue diamonds with black edges   (only if include_est)
        MSA-SS (true)   green dots with black edges

    The markers are drawn in that order, so the true point is always on top and
    never hidden by the others. A thin grey vertical line joins the lowest and
    highest marker of each row, so the spread between methods at a row reads as
    one stroke. Grey/white bands every 5 sites and minor y grid lines (at
    2..9 x 10^k) make individual rows and values easier to read off.

    include_est : bool
        True  -> MSA-SS, IDA-FX and the corrected estimate.
        False -> MSA-SS and IDA-FX only.
    """
    x = np.arange(len(df))
    # (column, legend label, colour, marker, extra plot kwargs) in drawing order.
    series = [("mafe_ida", "IDA-FX", IDA_COLOUR, "s", dict(mfc="none", mew=1.2, ms=5.5, zorder=3))]
    if include_est:
        series.append(("mafe_est", "MSA-SS est.", EST_COLOUR, "D",
                       dict(mfc=EST_COLOUR, mec="k", mew=0.6, ms=5, zorder=4)))
    series.append(("mafe_true", "MSA-SS (true)", TRUE_COLOUR, "o",
                   dict(mfc=TRUE_COLOUR, mec="k", mew=0.6, ms=4.5, zorder=5)))

    values = df[[s[0] for s in series]].to_numpy()

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.vlines(x, values.min(axis=1), values.max(axis=1), color="0.6", lw=0.8, zorder=1)
    for col, label, colour, marker, kw in series:
        ax.plot(x, df[col], ls="none", marker=marker, color=colour, label=label, **kw)

    # Log axis first, so the minor ticks (and their grid lines) are the log ones.
    ax.set_yscale("log")
    _site_axis_frame(ax, df, r"$\lambda_C$ (1/yr)")
    _five_site_bands(ax, df)
    ax.yaxis.grid(True, which="major", ls="-.", color="0.7", lw=0.8)
    ax.yaxis.grid(True, which="minor", ls=":", color="0.8", lw=0.6)

    # Legend above the axes: inside it would cover points near the bottom.
    ax.legend(ncol=len(series), loc="upper center", bbox_to_anchor=(0.5, 1.13),
              fontsize=9, frameon=False)
    _finish(fig, fp, dpi, plot_show)
    return fig, ax


def plot_mafe_ci_by_site(df, n_storeys, fp=None, level=0.95, dpi=300, plot_show=False):
    """MAFC with confidence-interval bars for one storey group, x = site number.

    At each site the three methods are offset sideways ('jittered') so their bars
    do not overlap: IDA-FX 0.25 to the left, MSA-SS (true) in the centre, the
    corrected MSA-SS estimate 0.25 to the right. Bars run from ci_lo_{m} to
    ci_hi_{m} (from `mafe_confidence_intervals`), symmetric on the log axis.
    Grey/white bands every 5 sites and minor y grid lines help read off rows and
    values; the title gives the number of rows whose IDA and estimate lie inside
    the MSA-SS interval.
    """
    sub = df.xs(n_storeys, level="n_storeys", drop_level=False)
    sites = sub.index.get_level_values("site").to_numpy()
    # (method, legend label, colour, marker, extra kwargs, sideways offset)
    series = [
        ("ida", "IDA-FX", IDA_COLOUR, "s", dict(mfc="w", mew=1.2, ms=5), -0.25),
        ("true", "MSA-SS (true)", TRUE_COLOUR, "o", dict(mfc=TRUE_COLOUR, mec="k", mew=0.6, ms=5), 0.0),
        ("est", "MSA-SS est.", EST_COLOUR, "D", dict(mfc=EST_COLOUR, mec="k", mew=0.6, ms=4.5), 0.25),
    ]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    # Grey band for sites 0-4, 10-14, ...; here x is the site number itself.
    for s in sites[(sites // 5) % 2 == 0]:
        ax.axvspan(s - 0.5, s + 0.5, color="0.92", lw=0, zorder=0)
    for m, label, colour, marker, kw, offset in series:
        y = sub[f"mafe_{m}"].to_numpy()
        yerr = [y - sub[f"ci_lo_{m}"].to_numpy(), sub[f"ci_hi_{m}"].to_numpy() - y]
        ax.errorbar(sites + offset, y, yerr=yerr, fmt=marker, color=colour, ecolor=colour,
                    elinewidth=1, capsize=2, label=label, zorder=3, **kw)

    ax.set_yscale("log")
    ax.set_xlim(sites.min() - 0.7, sites.max() + 0.7)
    ax.set_xticks(np.arange(0, sites.max() + 1, 5))
    ax.set_xticks(sites, minor=True)
    ax.set_xlabel(f"Site ({n_storeys}-storey)")
    ax.set_ylabel(r"$\lambda_C$ (1/yr)")
    ax.yaxis.grid(True, which="major", ls="-.", color="0.7", lw=0.8)
    ax.yaxis.grid(True, which="minor", ls=":", color="0.8", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor("w")
    ax.set_title(f"MAFC with {level:.0%} CI, {n_storeys}-storey   (inside MSA-SS {level:.0%} CI: "
                 f"IDA-FX {sub['ida_in_ci'].sum()}/{len(sub)}, MSA-SS est. {sub['est_in_ci'].sum()}/{len(sub)})",
                 fontsize=10)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17), fontsize=9, frameon=False)
    _finish(fig, fp, dpi, plot_show)
    return fig, ax


def plot_assessment_figures(df, fig_dir, dpi=300, plot_show=False,
                            duration_covariate=DURATION_COVARIATE,
                            duration_label=DURATION_COVARIATE_LABEL):
    """Produce and save every assessment figure into `fig_dir`.

    Saved files:
        var_pct_{source}_{term}.png  (12)   % share of each variance source
        var_abs_{source}_{term}.png  (12)   absolute contribution of each source
        var_total_{term}.png         (3)    terms of the total variance
        var_source_shares.png               stacked share by source (V, T, F, C)
        mafe_ln_ratio.png                   ln(lambda/lambda_true), IDA and estimate
        mafe_improvement.png                Delta, blue/orange by sign
        mafe_improvement_by_duration.png    Delta, coloured by the duration covariate
        mafe_improvement_by_design_group.png  Delta, grouped by design group
        mafe_parity.png                     predicted vs true MAFC
        mafe_parity_ida.png                 the same, IDA-FX only
        mafe_ratio_vs_duration.png          lambda/lambda_true against the duration ratio
        mafe_ratio_vs_duration_ida.png      the same, IDA-FX only
        mafe_correction_applied_vs_needed_ln.png
                                            correction needed vs applied (log), OLS fit
        mafe_z_calibration.png              standardised errors vs N(0, 1)
    """
    fig_dir = Path(fig_dir)
    plot_variance_component_sets(df, fig_dir, dpi=dpi, plot_show=plot_show)
    plot_variance_source_shares(df, fig_dir / "var_source_shares.png", dpi=dpi, plot_show=plot_show)
    plot_ln_mafe_ratio(df, fig_dir / "mafe_ln_ratio.png", dpi=dpi, plot_show=plot_show)
    plot_mafe_improvement(df, fig_dir / "mafe_improvement.png", dpi=dpi, plot_show=plot_show)
    plot_mafe_improvement(df, fig_dir / "mafe_improvement_by_duration.png",
                          colour_by=duration_covariate, colour_label=duration_label,
                          dpi=dpi, plot_show=plot_show)
    plot_mafe_improvement_by_design_group(df, fig_dir / "mafe_improvement_by_design_group.png",
                                          dpi=dpi, plot_show=plot_show)
    plot_mafe_parity(df, fig_dir / "mafe_parity.png", dpi=dpi, plot_show=plot_show)
    plot_mafe_parity(df, fig_dir / "mafe_parity_ida.png", include_est=False, dpi=dpi, plot_show=plot_show)
    plot_mafe_ratio_vs_duration(df, fig_dir / "mafe_ratio_vs_duration.png",
                                duration_covariate=duration_covariate, dpi=dpi, plot_show=plot_show)
    plot_mafe_ratio_vs_duration(df, fig_dir / "mafe_ratio_vs_duration_ida.png", include_est=False,
                                duration_covariate=duration_covariate, dpi=dpi, plot_show=plot_show)
    plot_mafe_correction_calibration(df, fig_dir / "mafe_correction_applied_vs_needed_ln.png",
                                     dpi=dpi, plot_show=plot_show)
    plot_z_calibration(df, fig_dir / "mafe_z_calibration.png", dpi=dpi, plot_show=plot_show)
