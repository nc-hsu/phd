"""Diagnostic plots for the fitted fragility-ratio models (the A-series fits).

What this module is for
-----------------------
Every function here takes one or more fitted models from
:mod:`phd_project.scripts.fragility_data_models` (``fm``) and draws one kind of
diagnostic plot:

    plot_conditional_effect       fitted line for one covariate, others at their mean,
                                  with partial residuals, a confidence band and a
                                  prediction band
    plot_observed_vs_fitted       exp(y_i) against exp(x_i' beta_hat), with the 1:1 line
    plot_variance_decomposition   stacked bars of tau_d^2 + tau_s^2 + tau_u^2 per model
    plot_fitted_forest            per-row forest plot: observed + CI against fitted + PI
    plot_whitened_residuals       QQ plot, r* vs fitted, r* vs a covariate
    plot_whitened_residuals_vs_covariates
                                  r* vs each of several covariates, one panel each,
                                  noting which are in the model
    whitened_residuals            the numbers behind the last two plots, as a DataFrame

Three properties hold for all of them.

1. **Univariate or joint, it does not matter.** A univariate :class:`fm.ModelFit`
   (one outcome, e.g. ``fits_theta["A3c_hb"]``) and a joint
   :class:`fm.JointModelFit` (both outcomes, e.g. ``joint["A8e_bc_hb"]``) are both
   accepted. Every function asks for ONE ``quantity`` ("theta" or "beta"); for a
   joint fit that picks out that outcome's block, for a univariate fit it must
   match the fit's own quantity (or be left as None). The translation is done in
   one place, :func:`fit_view`, so the plotting code never branches on the fit
   type.

2. **Several models on one axes.** Wherever a function takes ``models`` it
   accepts either a single fit or a list of fits, and overlays them, so rungs of
   the ladder can be compared directly (e.g. A8e vs A8f on one observed-vs-fitted
   plot). With more than one model the default colours switch from "by storey"
   to "by model" (see ``colour_by`` below) so the models can be told apart.

3. **Everything drawn is restyleable** through two arguments that every plot
   function shares, ``style`` and ``model_styles``. See "Styling" below.

Styling
-------
Each plot is built from named ELEMENTS (``"points"``, ``"line"``, ``"ci_band"``
...); each function's docstring lists its elements and which matplotlib call
draws them. An element's keyword arguments are assembled in layers, later
layers winning:

    1. the built-in defaults (the look of the original nb 073 plots), including
       the automatic colour/marker for the storey or model being drawn;
    2. ``style[element]``                          -- every model, every storey;
    3. ``style["storeys"][n_storeys][element]``    -- one storey, every model;
    4. ``model_styles[k][element]``                -- one model, every storey;
    5. ``model_styles[k]["storeys"][n_storeys][element]`` -- one model, one storey.

So ``style`` is a nested dictionary ``{element: {kwarg: value}}``, with an
optional ``"storeys"`` entry one level deeper, and ``model_styles`` is a list of
such dictionaries, one per model in the order the models were passed (or a dict
keyed by ``model_id``). Any keyword the matplotlib call accepts can be given,
under any of its aliases (``lw`` / ``linewidth``, ``ls`` / ``linestyle`` ...);
they are normalised before merging so an alias can never clash with a default.

Two conventions on top of plain keywords:

* **None hides an element.** ``style={"pi_band": None}`` draws no prediction
  band; ``model_styles=[{}, {"points": None}]`` draws the second model's line but
  not its points; ``style={"storeys": {5: None}}`` drops the 5-storey rows.
* **Labels are templates.** A ``"label"`` value is formatted with the fields
  ``{model}`` (model_id), ``{ns}`` (number of storeys), ``{level}`` (e.g. "95%")
  and, where it makes sense, ``{n_out}`` and ``{n}``. E.g.
  ``style={"points": {"label": "{model}: {ns}s"}}``.

For filled markers and bands, ``color`` (or ``c``) means the FILL colour; use
``edgecolor`` for the outline.

The automatic colours and markers come from four palette entries, which can
also be put in ``style`` (they are not elements, just lists/dicts):

    style["storey_colours"]  {n_storeys: colour}, default STOREY_COLOURS
    style["storey_markers"]  {n_storeys: marker}, default STOREY_MARKERS
    style["model_colours"]   [colour, ...] cycled over models, default MODEL_COLOURS
    style["model_markers"]   [marker, ...] cycled over models, default MODEL_MARKERS

and ``colour_by`` chooses how they are used:

    "storey"  colour = storey colour, marker = model marker
    "model"   colour = model colour,  marker = storey marker
    "auto"    (default) "storey" for one model, "model" for several

Example
-------
>>> import phd_project.scripts.fragility_data_models_diagnostics as fmd
>>> fig, ax = plt.subplots()
>>> fmd.plot_observed_vs_fitted(
...     ax, [joint["A8e_bc_hb"], joint["A8f_bc_hb"]], "theta",
...     style={"identity": {"color": "0.3", "lw": 1.5},
...            "points": {"alpha": 0.7}},
...     model_styles=[{"points": {"marker": "o"}}, {"points": {"marker": "x"}}])
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cbook
from matplotlib import collections as mcoll
from matplotlib import lines as mlines
from matplotlib import patches as mpatches
from matplotlib import text as mtext
from scipy import stats
from scipy.linalg import solve_triangular

import phd_project.scripts.fragility_data_models as fm


# =============================================================================
# Palettes and constants
# =============================================================================

#: Default colour per number of storeys (the original nb 073 choice: red/blue).
STOREY_COLOURS: dict[int, str] = {3: "r", 5: "b"}
#: Default marker per number of storeys, used when colour encodes the MODEL.
STOREY_MARKERS: dict[int, str] = {3: "o", 5: "s"}
#: Colours cycled over models when colour encodes the MODEL (Okabe-Ito, which
#: stays distinguishable for the common forms of colour-blindness).
MODEL_COLOURS: list[str] = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
                            "#E69F00", "#56B4E9", "#000000"]
#: Markers cycled over models when colour encodes the STOREY. The first is "o",
#: so a single model looks exactly like the original plots.
MODEL_MARKERS: list[str] = ["o", "s", "^", "D", "v", "P", "X"]
#: Markers of the fitted values in the forest plot when several models are
#: told apart by shape; no square, which marks the observed value there.
FITTED_MARKERS: list[str] = ["D", "v", "^", "P", "X", "<", ">"]
#: Fallback colours for a storey count not in STOREY_COLOURS.
_EXTRA_STOREY_COLOURS: list[str] = ["g", "m", "c", "y"]

#: Variance levels in the order they are stacked, with the colour of each
#: segment in :func:`plot_variance_decomposition`. The names are the keys the
#: joint fits use for ``T`` and the names :class:`fm.CovarianceDecomposition`
#: uses for its parts.
LEVELS: tuple[str, ...] = ("design", "site", "row")
LEVEL_COLOURS: dict[str, str] = {"design": "#0072B2", "site": "#E69F00", "row": "#009E73"}
#: Univariate ModelFit.tau2 keys for each level.
_UNIVARIATE_TAU_KEYS: dict[str, str] = {"design": "tau_d2", "site": "tau_s2", "row": "tau_u2"}

#: Position of each outcome in the 2x2 matrices of a joint fit (T, T_total).
_Q_INDEX: dict[str, int] = {q: i for i, q in enumerate(fm.OUTCOMES)}

# The matplotlib artist each "kind" of element is drawn as. Used only to
# normalise keyword aliases (lw -> linewidth ...) before merging, so a user's
# alias replaces the default instead of being passed alongside it.
_ARTISTS = {
    "scatter": mcoll.PathCollection,    # ax.scatter
    "line": mlines.Line2D,              # ax.plot, ax.axhline, ax.axline
    "fill": mcoll.PolyCollection,       # ax.fill_between
    "hlines": mcoll.LineCollection,     # ax.hlines
    "bar": mpatches.Rectangle,          # ax.barh
    "text": mtext.Text,                 # ax.text
    "dict": None,                       # passed to something that is not an artist
}                                       # (ax.legend, ax.grid): merged, not normalised
# Kinds whose "color" should mean the FILL colour. For ax.scatter and ax.barh an
# explicit facecolor silently beats color, so a user's color would be ignored
# if the default were written as facecolor; translating it avoids that trap.
_FILL_KINDS = {"scatter", "bar", "fill"}

_NO_STOREY = object()   # sentinel: "this element is not split by storey"


def ratio_label(quantity: str, prefix: str = "") -> str:
    r"""Axis label for the ratio of one quantity, e.g. ``$\theta_{SS}\,/\,\theta_{FX}$``.

    ``prefix`` is prepended with a space, e.g. ``ratio_label("beta", "Fitted")``.
    """
    lab = rf"$\{quantity}_{{SS}}\,/\,\{quantity}_{{FX}}$"
    return f"{prefix} {lab}" if prefix else lab


# =============================================================================
# Uniform view of one outcome of one fit
# =============================================================================


class FitView:
    """Everything a diagnostic plot needs about ONE outcome of ONE fit.

    A univariate :class:`fm.ModelFit` and a joint :class:`fm.JointModelFit` store
    the same information in different shapes: the joint fit stacks theta over
    beta, so its design matrix is block-diagonal ``blockdiag(X_theta, X_beta)``,
    its coefficient vector holds both outcomes' coefficients, and its total
    covariance ``Sigma`` is 2n x 2n. This class cuts out the block for one
    outcome so that, from here on, a joint fit looks exactly like a univariate
    one. That is legitimate for every plot in this module because each one shows
    a single outcome, and for a single outcome only its own block matters:

    * fitted values: ``X @ beta`` of a block-diagonal X is the two blocks'
      ``X_q @ beta_q`` stacked, so taking the block first gives the same numbers;
    * ``Var(x' beta_hat)`` only involves this outcome's coefficients;
    * the MARGINAL covariance of one outcome is its diagonal block of Sigma
      (Gelman et al. BDA3 [2], Ch. 3.5, ~pp. 70-73, approximate: the marginal of a
      multivariate normal is the sub-vector with the sub-matrix of the covariance).

    Build it with :func:`fit_view` rather than directly.

    Attributes
    ----------
    fit : ModelFit or JointModelFit
        The fit this view was cut from.
    model_id, quantity : str
    joint : bool
        True when ``fit`` is a joint fit.
    contrast : fm.Contrast
        The UNIVARIATE contrast of this outcome: observed ``y_hat``, replicates
        ``y``, factor/covariate table ``rows`` and sampling covariance ``V``.
    X : ndarray (n, p)
        This outcome's design matrix: intercept plus centred and scaled
        covariates, built by the same code the fit used.
    x_names : list of str
        Column names of ``X`` ("intercept" first).
    beta : ndarray (p,)
        This outcome's coefficients.
    vcov : ndarray (p, p)
        Covariance of ``beta``; see :attr:`vcov_source`.
    vcov_source : {"bootstrap", "model", "model (diagonal)"}
        Where ``vcov`` came from. The bootstrap covariance is preferred when the
        fit has replicates holding every coefficient: on a "_hb" (hybrid) fit it
        is the superpopulation covariance and also reflects the re-estimated
        tau's, which the model-based one treats as known. Otherwise the
        model-based ``(X' Sigma^-1 X)^-1`` is used; for a DerSimonian-Laird fit
        only its diagonal (``se_beta**2``) exists.
    scaling : dict
        ``{covariate: {"mean", "sd"}}``, the centring and scaling of each column
        of ``X``.
    tau2 : dict
        ``{"design", "site", "row"}`` -> variance of this outcome's true effects at
        that level, 0.0 for a level the model does not have.
    """

    def __init__(self, fit, quantity: str | None = None):
        self.fit = fit
        self.model_id = fit.model_id
        # Duck-typed, not isinstance: with %autoreload a reloaded fm defines a
        # new JointModelFit class, and fits made before the reload would then
        # fail an isinstance check against it.
        self.joint = hasattr(fit, "T_total") and hasattr(fit.contrast, "theta")

        if self.joint:
            if quantity not in _Q_INDEX:
                raise ValueError(f"{self.model_id} is a joint fit: pass quantity='theta' or "
                                 f"'beta', got {quantity!r}")
            self.quantity = quantity
            self._init_joint()
        else:
            if quantity is not None and quantity != fit.quantity:
                raise ValueError(f"{self.model_id} is a {fit.quantity} fit, not {quantity}")
            self.quantity = fit.quantity
            self._init_univariate()

    # -- construction, one per fit type -------------------------------------

    def _init_joint(self) -> None:
        fit, q = self.fit, self.quantity
        self.qi = _Q_INDEX[q]
        self.contrast = getattr(fit.contrast, q)          # the univariate Contrast of q
        n = self.contrast.n
        # The full block-diagonal X, rebuilt with the fit's own centring/scaling.
        # Rows: theta block (first n) then beta block (last n); columns: the
        # coefficients in fit.x_names order, as (outcome, name) pairs.
        X, x_names, _ = fm.build_X_joint(fit.spec, fit.contrast)
        cols = [k for k, (oq, _) in enumerate(x_names) if oq == q]
        rows = slice(0, n) if self.qi == 0 else slice(n, 2 * n)
        self._cols = cols                                  # positions in the joint vector
        self.X = X[rows][:, cols]
        self.x_names = [x_names[k][1] for k in cols]
        self.beta = np.asarray(fit.beta, dtype=float)[cols]
        Vb_boot = fit.vcov_beta_boot
        if Vb_boot is not None:
            self.vcov, self.vcov_source = Vb_boot[np.ix_(cols, cols)], "bootstrap"
        else:
            self.vcov, self.vcov_source = fit.vcov_beta[np.ix_(cols, cols)], "model"
        self.scaling = fit.scaling[q]
        # T[level] is 2x2 (theta, beta); [qi, qi] is this outcome's variance.
        self.tau2 = {lev: float(fit.T[lev][self.qi, self.qi]) if lev in fit.T else 0.0
                     for lev in LEVELS}

    def _init_univariate(self) -> None:
        fit = self.fit
        self.qi = None
        self.contrast = fit.contrast
        X, x_names, scaling = fm.build_X(fit.spec, self.contrast)
        self.X, self.x_names, self.scaling = X, list(x_names), scaling
        self.beta = np.asarray(fit.beta, dtype=float)
        self.vcov, self.vcov_source = self._univariate_vcov()
        self.tau2 = {lev: float(fit.tau2.get(key, 0.0))
                     for lev, key in _UNIVARIATE_TAU_KEYS.items()}

    def _univariate_vcov(self) -> tuple[np.ndarray, str]:
        """Coefficient covariance of a univariate fit, bootstrap first."""
        fit = self.fit
        reps = fit.replicates
        if reps is not None:
            cols = []
            for nm in self.x_names:
                col = fm.beta_column(nm)
                if col not in reps and nm == "intercept" and "M_star" in reps:
                    col = "M_star"     # older replicate frames carry only the intercept, as M_star
                cols.append(col)
            if all(c in reps for c in cols):
                V = np.atleast_2d(np.cov(reps[cols].to_numpy(float), rowvar=False))
                return V, "bootstrap"
        if "vcov_beta" in fit.raw:
            return np.asarray(fit.raw["vcov_beta"], dtype=float), "model"
        # DerSimonian-Laird keeps only the standard errors.
        return np.diag(np.asarray(fit.se_beta, dtype=float) ** 2), "model (diagonal)"

    # -- derived quantities ---------------------------------------------------

    @property
    def n(self) -> int:
        return self.contrast.n

    @property
    def tau2_total(self) -> float:
        """``tau_d^2 + tau_s^2 + tau_u^2``: total between-row variance of the true effects."""
        return float(sum(self.tau2.values()))

    @property
    def index(self) -> pd.Index:
        """Row index of this outcome: (site, n_storeys) at row scope."""
        return self.contrast.y_hat.index

    @property
    def y_obs(self) -> np.ndarray:
        """Observed log-ratios y_i."""
        return self.contrast.y_hat.to_numpy(float)

    @property
    def v(self) -> np.ndarray:
        """Sampling variance of each row, the diagonal of the contrast's V."""
        return np.diag(self.contrast.V)

    @cached_property
    def fitted(self) -> np.ndarray:
        """Fitted log-ratios ``x_i' beta_hat``."""
        return self.X @ self.beta

    @cached_property
    def var_fitted(self) -> np.ndarray:
        """``Var(x_i' beta_hat) = x_i' vcov x_i`` for every row.

        ``einsum`` takes the diagonal of ``X vcov X'`` without building the
        n x n matrix.
        """
        return np.einsum("ij,jk,ik->i", self.X, self.vcov, self.X)

    @cached_property
    def Sigma(self) -> np.ndarray:
        """(n, n) total marginal covariance of this outcome's effect sizes.

        ``Sigma = V_known + sum_l tau_l^2 G_l`` -- the matrix the REML fit
        weighted by (see :class:`fm.CovarianceDecomposition`). For a joint fit,
        its diagonal block of ``raw["Sigma"]``; ``kron(T_l, G_l)`` puts
        ``T_l[0,0] G_l`` top-left, so theta is ``[:n, :n]`` and beta ``[n:, n:]``.
        For a univariate fit, rebuilt by :func:`fm.marginal_covariance`, which
        also works for DL fits (which keep no Sigma of their own).
        """
        if self.joint:
            S = np.asarray(self.fit.raw["Sigma"], dtype=float)
            n = self.n
            return S[:n, :n] if self.qi == 0 else S[n:, n:]
        return fm.marginal_covariance(self.fit).Sigma

    @cached_property
    def storeys(self) -> np.ndarray:
        """Number of storeys per row, or all None when the rows have none.

        Read from the "n_storeys" index level at row scope; at group scope (A2,
        51 design groups) from the factor table if it has the column.
        """
        idx = self.index
        if "n_storeys" in (idx.names or []):
            return np.asarray(idx.get_level_values("n_storeys"))
        if "n_storeys" in self.contrast.rows.columns:
            return self.contrast.rows["n_storeys"].to_numpy()
        return np.full(self.n, None, dtype=object)

    def slope(self, covariate: str) -> float:
        """Fitted slope (per SD) of ``covariate``; 0.0 if the model does not include it."""
        return float(self.beta[self.x_names.index(covariate)]) if covariate in self.x_names else 0.0

    def covariate_values(self, name: str) -> np.ndarray:
        """Raw values of a covariate column of the rows, or of an index level."""
        rows = self.contrast.rows
        if name in rows.columns:
            return rows[name].to_numpy()
        if name in (self.index.names or []):
            return np.asarray(self.index.get_level_values(name))
        raise KeyError(f"{name!r} is neither a column of the {self.quantity} rows nor an "
                       f"index level {list(self.index.names)}")

    def covariate_scaling(self, covariate: str) -> tuple[float, float]:
        """(mean, sd) used to centre/scale a covariate.

        The fit's own values when the covariate is in the model; otherwise the
        same recipe :func:`fm.build_X` would use (mean, SD with ddof=1), so a
        covariate the model left out can still be placed on a common axis.
        """
        if covariate in self.scaling:
            sc = self.scaling[covariate]
            return float(sc["mean"]), float(sc["sd"])
        raw = self.covariate_values(covariate).astype(float)
        return float(np.mean(raw)), float(np.std(raw, ddof=1))

    def __repr__(self) -> str:
        return (f"FitView({self.model_id!r}, {self.quantity!r}, n={self.n}, "
                f"p={len(self.x_names)}, vcov={self.vcov_source})")


def fit_view(model, quantity: str | None = None) -> FitView:
    """Return a :class:`FitView` of one outcome of ``model`` (a FitView passes through)."""
    if isinstance(model, FitView):
        if quantity is not None and quantity != model.quantity:
            raise ValueError(f"view is of {model.quantity}, not {quantity}")
        return model
    return FitView(model, quantity)


def _as_views(models, quantity: str | None) -> tuple[list[FitView], bool]:
    """Normalise ``models`` (one fit or a sequence of fits) to a list of views.

    The second value says whether a single fit was passed, so functions that
    return per-model results can return a bare object in that case.
    """
    single = not isinstance(models, (list, tuple))
    seq = [models] if single else list(models)
    if not seq:
        raise ValueError("no models given")
    return [fit_view(m, quantity) for m in seq], single


# =============================================================================
# Style resolution
# =============================================================================


class _Fmt(dict):
    """Format mapping that leaves unknown {fields} in place instead of raising."""

    def __missing__(self, key):
        return "{" + key + "}"


def _normalise(kind: str, kw: Mapping) -> dict:
    """Canonical matplotlib keyword names for one layer of an element's style."""
    kw = dict(kw)
    if kind in _FILL_KINDS:
        # "color"/"c" mean the fill colour; an explicit facecolor in the SAME
        # layer still wins.
        for alias in ("color", "c"):
            if alias in kw:
                val = kw.pop(alias)
                kw.setdefault("facecolor", val)
    artist = _ARTISTS[kind]
    return cbook.normalize_kwargs(kw, artist) if artist is not None else kw


class _Styler:
    """Resolve the keyword arguments of every element of one plot.

    Holds the user's ``style`` and ``model_styles`` and the colour scheme, and
    answers "what keywords does element E get for model k (and storey ns)?" by
    stacking the layers described in the module docstring. A return value of
    None means the element is hidden.
    """

    def __init__(self, style: Mapping | None, model_styles, views: Sequence[FitView],
                 colour_by: str = "auto"):
        self.style = dict(style or {})
        self.views = views
        self.K = len(views)
        if colour_by == "auto":
            colour_by = "storey" if self.K == 1 else "model"
        if colour_by not in ("storey", "model"):
            raise ValueError(f"colour_by must be 'auto', 'storey' or 'model', got {colour_by!r}")
        self.colour_by = colour_by

        # model_styles: list aligned with views, or {model_id: dict}.
        if model_styles is None:
            self.model_styles = [{} for _ in views]
        elif isinstance(model_styles, Mapping):
            self.model_styles = [model_styles.get(v.model_id, {}) for v in views]
        else:
            ms = list(model_styles)
            if len(ms) != self.K:
                raise ValueError(f"model_styles has {len(ms)} entries for {self.K} models")
            self.model_styles = ms

        self.storey_colours = {**STOREY_COLOURS, **self.style.get("storey_colours", {})}
        self.storey_markers = {**STOREY_MARKERS, **self.style.get("storey_markers", {})}
        self.model_colours = list(self.style.get("model_colours", MODEL_COLOURS))
        self.model_markers = list(self.style.get("model_markers", MODEL_MARKERS))

    # -- palette lookups ------------------------------------------------------

    def storey_colour(self, ns) -> str:
        if ns in self.storey_colours:
            return self.storey_colours[ns]
        if ns is None:                      # rows without a storey count
            return "k"
        # A storey count with no assigned colour: a spare one, the same every time.
        return _EXTRA_STOREY_COLOURS[int(ns) % len(_EXTRA_STOREY_COLOURS)]

    def storey_marker(self, ns) -> str:
        return self.storey_markers.get(ns, "o")

    def model_colour(self, k: int) -> str:
        return self.model_colours[k % len(self.model_colours)]

    def model_marker(self, k: int) -> str:
        return self.model_markers[k % len(self.model_markers)]

    def point_colour(self, k: int, ns) -> str:
        """Automatic colour of a per-storey element of model k."""
        if self.colour_by == "storey" and ns is not None:
            return self.storey_colour(ns)
        return self.model_colour(k)

    def point_marker(self, k: int, ns) -> str:
        """Automatic marker of a per-storey element of model k."""
        if self.colour_by == "model" and ns is not None:
            return self.storey_marker(ns)
        return self.model_marker(k)

    def series_colour(self, k: int, default: str) -> str:
        """Colour of a per-model element that is not split by storey (a line, a band).

        One model keeps the element's own default (e.g. a black fitted line);
        several models need the model colour to be told apart.
        """
        return default if self.K == 1 else self.model_colour(k)

    # -- layered lookup -------------------------------------------------------

    @staticmethod
    def _get(d, element):
        """``d[element]``: {} if absent, None if explicitly hidden."""
        if d is None:
            return None
        return d.get(element, {})

    @staticmethod
    def _storey(d, ns):
        """``d["storeys"][ns]``: {} if absent, None if that storey is hidden."""
        if d is None:
            return None
        storeys = d.get("storeys", {}) or {}
        return storeys.get(ns, {})

    def kw(self, element: str, kind: str, defaults: Mapping, *, k: int = 0,
           ns=_NO_STOREY, fmt: Mapping | None = None) -> dict | None:
        """Final keywords of ``element`` for model ``k`` (and storey ``ns``), or None to hide.

        ``kind`` is the matplotlib call it goes to (see ``_ARTISTS``);
        ``fmt`` holds extra fields for the label template.
        """
        mstyle = self.model_styles[k] if k < len(self.model_styles) else {}
        layers = [defaults, self._get(self.style, element)]
        if ns is not _NO_STOREY:
            layers.append(self._get(self._storey(self.style, ns), element))
        layers.append(self._get(mstyle, element))
        if ns is not _NO_STOREY:
            layers.append(self._get(self._storey(mstyle, ns), element))

        out: dict = {}
        for layer in layers:
            if layer is None:
                return None
            if layer:
                out.update(_normalise(kind, layer))

        lab = out.get("label")
        if isinstance(lab, str):
            fields = {"model": self.views[k].model_id if self.views else "",
                      "ns": "" if ns is _NO_STOREY else ns}
            fields.update(fmt or {})
            out["label"] = lab.format_map(_Fmt(fields))
        return out


def _storey_groups(view: FitView):
    """Yield ``(ns, mask)`` per storey count, in ascending order (None last)."""
    storeys = view.storeys
    keys = sorted({s for s in storeys if s is not None})
    if any(s is None for s in storeys):
        keys.append(None)
    for ns in keys:
        mask = np.array([s == ns for s in storeys]) if ns is None else np.asarray(storeys == ns)
        yield ns, mask


def _fitted_marker(st: _Styler, k: int) -> str:
    """Marker of model k's fitted diamonds in the forest plot.

    Always a diamond when the diamonds are told apart by colour (one model, or
    colour by model). When colour encodes the storey, the models need different
    shapes; FITTED_MARKERS avoids the square, which is the observed point.
    """
    if st.K == 1 or st.colour_by == "model":
        return "D"
    return FITTED_MARKERS[k % len(FITTED_MARKERS)]


def _storey_text(ns) -> str:
    return "all rows" if ns is None else f"{ns}-storey"


def _level(alpha: float) -> str:
    """'95%' for alpha = 0.05."""
    return f"{1 - alpha:.0%}"


# =============================================================================
# Conditional effect
# =============================================================================


def plot_conditional_effect(ax, models, quantity: str | None, covariate: str,
                            significance_level: float = 0.05, vline_at_one: bool = True,
                            *, exp_x: bool = True, size_range: tuple[float, float] = (7.5, 142.5),
                            style: Mapping | None = None, model_styles=None,
                            colour_by: str = "auto"):
    """Conditional-effect plot of one covariate, every other covariate at its mean.

    On the log scale the fitted line is

        y_hat(x_j) = m + gamma_j * (x_j - mean_j) / sd_j

    with ``m`` the intercept (the fit at the covariate means, because the
    covariates are centred) and ``gamma_j`` the slope per SD. It is drawn on the
    RATIO scale, ``exp(y_hat)`` against ``exp(x_j)`` (the covariates are logs of
    ratios or of physical quantities, so ``exp`` puts them back in natural units).

    The points are PARTIAL RESIDUALS: each row's observed y with the fitted
    effect of its OTHER covariates removed,

        y*_i = y_i - sum_{k != j} gamma_k z_ik,     z_ik = (x_ik - mean_k) / sd_k,

    so each point sits where it would if those covariates were at their means,
    on the same footing as the line (Chatterjee & Hadi [7], Ch. 4, sec. 4.12,
    ~pp. 108-112, approximate).

    The CONFIDENCE band is for the mean line. Only the intercept and slope_j move
    along it, so

        Var(y_hat) = Var(m) + 2 z Cov(m, gamma_j) + z^2 Var(gamma_j),

    which is narrowest at the covariate mean (z = 0). The PREDICTION band adds
    the total between-row variance tau_d^2 + tau_s^2 + tau_u^2, so it is where
    the true effect of a NEW site/design is expected (Borenstein et al. [1],
    Ch. 17, ~pp. 127-133, approximate).

    Point size follows the weight ``1 / (v_i + tau_tot^2)``: larger is more
    precise.

    A model that does NOT include ``covariate`` is still drawn: its slope is 0,
    so its line is flat at its intercept and its band carries only Var(m). That
    is what the model says, and is useful next to a model that does include it.

    Parameters
    ----------
    ax : matplotlib Axes
    models : fit or list of fits
        ModelFit and/or JointModelFit, overlaid in order.
    quantity : {"theta", "beta"} or None
        Required for joint fits; None takes a univariate fit's own quantity.
    covariate : str
        Column of the contrast rows, e.g. "d_ln_ds595_mean_unif".
    significance_level : float
        alpha; both bands are (1 - alpha).
    vline_at_one : bool
        Draw the vertical reference line at x = 1 (a covariate ratio of one,
        e.g. "same duration as the fixed set"). Same as ``style={"vline": None}``
        when False.
    exp_x : bool
        Plot ``exp(covariate)`` (True) or the covariate as stored (False).
    size_range : (float, float)
        Smallest and largest marker area for the weight-scaled points. Giving
        ``"s"`` in the points style replaces the scaling with a fixed size.
    style, model_styles, colour_by
        See the module docstring.

    Elements
    --------
    pi_band   ax.fill_between   prediction band
    ci_band   ax.fill_between   confidence band
    line      ax.plot           fitted line
    points    ax.scatter        partial residuals, one call per storey
    hline     ax.axhline        y = 1 (no difference in fragility)
    vline     ax.axvline        x = 1

    Returns
    -------
    ax
    """
    views, _ = _as_views(models, quantity)
    st = _Styler(style, model_styles, views, colour_by)
    z_crit = stats.norm.ppf(1 - significance_level / 2)
    lev = _level(significance_level)
    tx = np.exp if exp_x else (lambda a: a)
    multi = st.K > 1

    for k, vw in enumerate(views):
        x_raw = vw.covariate_values(covariate).astype(float)
        mu_j, sd_j = vw.covariate_scaling(covariate)
        i_int = vw.x_names.index("intercept")
        m_q = vw.beta[i_int]
        gamma_j = vw.slope(covariate)

        # 2x2 covariance of (intercept, slope_j). For a covariate the model
        # does not have, the slope is fixed at 0 and so has no variance.
        if covariate in vw.x_names:
            idx = [i_int, vw.x_names.index(covariate)]
            Vb2 = vw.vcov[np.ix_(idx, idx)]
        else:
            Vb2 = np.array([[vw.vcov[i_int, i_int], 0.0], [0.0, 0.0]])

        # Line and bands on a grid over the observed range of the covariate.
        x_grid = np.linspace(x_raw.min(), x_raw.max(), 200)
        z_grid = (x_grid - mu_j) / sd_j                   # same scaling as the fit
        y_line = m_q + gamma_j * z_grid
        # Vb2 entries are scalars, z_grid is (200,): numpy broadcasts.
        var_ci = Vb2[0, 0] + 2 * z_grid * Vb2[0, 1] + z_grid ** 2 * Vb2[1, 1]
        var_pi = var_ci + vw.tau2_total                   # + tau_tot^2 -> new site/design
        ci_lo, ci_hi = y_line - z_crit * np.sqrt(var_ci), y_line + z_crit * np.sqrt(var_ci)
        pi_lo, pi_hi = y_line - z_crit * np.sqrt(var_pi), y_line + z_crit * np.sqrt(var_pi)

        # Partial residuals: remove every OTHER covariate's fitted contribution.
        y_part = vw.y_obs.copy()
        for cov_k in vw.x_names:
            if cov_k in ("intercept", covariate):
                continue
            z_k = (vw.covariate_values(cov_k).astype(float) - vw.scaling[cov_k]["mean"]) \
                / vw.scaling[cov_k]["sd"]
            y_part -= vw.slope(cov_k) * z_k

        # Weight-scaled marker area: larger = more precise.
        w = 1.0 / (vw.v + vw.tau2_total)
        span = w.max() - w.min()
        lo_s, hi_s = size_range
        size = np.full_like(w, (lo_s + hi_s) / 2) if span == 0 else \
            lo_s + (hi_s - lo_s) * (w - w.min()) / span

        band_c = st.series_colour(k, "C0")
        pre = "{model}: " if multi else ""
        kw = st.kw("pi_band", "fill", {"facecolor": band_c, "alpha": 0.12, "linewidth": 0,
                                       "label": pre + "{level} prediction band (new site/design)"},
                   k=k, fmt={"level": lev})
        if kw is not None:
            ax.fill_between(tx(x_grid), np.exp(pi_lo), np.exp(pi_hi), **kw)
        kw = st.kw("ci_band", "fill", {"facecolor": band_c, "alpha": 0.30, "linewidth": 0,
                                       "label": pre + "{level} confidence band (mean line)"},
                   k=k, fmt={"level": lev})
        if kw is not None:
            ax.fill_between(tx(x_grid), np.exp(ci_lo), np.exp(ci_hi), **kw)
        kw = st.kw("line", "line", {"color": st.series_colour(k, "k"), "linewidth": 2,
                                    "label": pre + "fitted, others at mean"}, k=k)
        if kw is not None:
            ax.plot(tx(x_grid), np.exp(y_line), **kw)

        for ns, sel in _storey_groups(vw):
            kw = st.kw("points", "scatter",
                       {"marker": st.point_marker(k, ns), "s": size[sel],
                        "facecolor": st.point_colour(k, ns), "edgecolor": "k",
                        "linewidth": 0.7, "alpha": 0.5,
                        "label": pre + f"{_storey_text(ns)} (partial residual)"},
                       k=k, ns=ns)
            if kw is not None:
                ax.scatter(tx(x_raw[sel]), np.exp(y_part[sel]), **kw)

    kw = st.kw("hline", "line", {"color": "0.5", "linewidth": 0.8, "linestyle": "--"})
    if kw is not None:
        ax.axhline(1.0, **kw)
    if vline_at_one:
        kw = st.kw("vline", "line", {"color": "0.5", "linewidth": 0.8, "linestyle": "--"})
        if kw is not None:
            ax.axvline(1.0, **kw)
    return ax


# =============================================================================
# Observed vs fitted
# =============================================================================


def plot_observed_vs_fitted(ax, models, quantity: str | None = None, *,
                            style: Mapping | None = None, model_styles=None,
                            colour_by: str = "auto"):
    """Observed ratio ``exp(y_i)`` (x) against fitted ratio ``exp(x_i' beta_hat)`` (y).

    Points on the 1:1 line are rows the fixed effects reproduce. The vertical
    scatter of the points around the line is NOT expected to vanish: the fixed
    effects only describe the mean, and each row also carries its own
    random effects (tau_d, tau_s, tau_u) and sampling error. A fitted value
    that barely moves while the observed values spread out means the covariates
    explain little of the between-row variation; a tilt away from 1:1 is the
    shrinkage of a regression towards the mean (Gelman & Hill [3], Ch. 3.6,
    ~pp. 48-49 and Ch. 12.2, ~pp. 252-256, approximate).

    Parameters
    ----------
    ax : matplotlib Axes
    models : fit or list of fits
    quantity : {"theta", "beta"} or None
    style, model_styles, colour_by
        See the module docstring.

    Elements
    --------
    identity  ax.axline   the 1:1 line
    points    ax.scatter  one call per model per storey

    Returns
    -------
    ax
    """
    views, _ = _as_views(models, quantity)
    st = _Styler(style, model_styles, views, colour_by)
    multi = st.K > 1

    kw = st.kw("identity", "line", {"color": "k", "linestyle": ":"})
    if kw is not None:
        ax.axline((1.0, 1.0), slope=1.0, **kw)    # infinite, so axis limits never cut it short

    for k, vw in enumerate(views):
        for ns, sel in _storey_groups(vw):
            kw = st.kw("points", "scatter",
                       {"marker": st.point_marker(k, ns), "s": 36,
                        "facecolor": st.point_colour(k, ns), "edgecolor": "k",
                        "linewidth": 1.0, "alpha": 0.5,
                        "label": ("{model}: " if multi else "") + _storey_text(ns)},
                       k=k, ns=ns)
            if kw is not None:
                ax.scatter(np.exp(vw.y_obs[sel]), np.exp(vw.fitted[sel]), **kw)
    return ax


# =============================================================================
# Variance decomposition
# =============================================================================


def plot_variance_decomposition(models, quantity: str, ax, show_labels: bool = True, *,
                                share_threshold: float = 0.12, labels: Sequence[str] | None = None,
                                style: Mapping | None = None):
    """Horizontal stacked bars: one bar per model, length = total between-row VARIANCE.

    Bar length is the total variance of the true effects (log scale),

        tau_tot^2 = tau_d^2 + tau_s^2 + tau_u^2,

    and each segment is one component's tau_k^2. Because variances add, the
    segments sum exactly to the bar and each segment's length IS that component
    -- no rescaling needed. (Standard deviations would not add:
    tau_d + tau_s + tau_u > tau_tot.) A level a model does not include (e.g.
    "site" in A1b, or "design" and "site" in a DL fit) is drawn as zero, so
    every bar has the same three segments and the models line up.

    Unlike the other plots, every model is already its own bar, so there is no
    ``model_styles``: styling is per LEVEL.

    Parameters
    ----------
    models : list of ModelFit / JointModelFit (can be mixed)
        Drawn top-to-bottom in list order.
    quantity : {"theta", "beta"}
    ax : matplotlib Axes
    show_labels : bool
        Write each segment's share of the variance inside it, and the total
        variance (plus its SD) at the end of the bar.
    share_threshold : float
        Only segments at least this share of their bar get a text label;
        thinner ones are still visible as colour.
    labels : list of str, optional
        y tick labels, one per model; default the model ids.
    style : dict, optional
        Nested dict of element overrides, as in the module docstring.

    Elements
    --------
    design, site, row   ax.barh    one segment per level ("height" sets bar thickness)
    segment_text        ax.text    the share inside each segment
    total_text          ax.text    total variance and tau_tot at the bar end
    legend              ax.legend  keywords for the legend (None hides it)
    grid                ax.grid    keywords for the grid (None hides it)

    Returns
    -------
    ax
    """
    if quantity not in _Q_INDEX:
        raise ValueError(f"quantity must be 'theta' or 'beta', got {quantity!r}")
    views, _ = _as_views(models, quantity)
    st = _Styler(style, None, views, "model")

    y_pos = np.arange(len(views))
    x_max = 0.0
    for y, vw in zip(y_pos, views):
        tau2 = vw.tau2
        total_var = sum(tau2.values())
        x_max = max(x_max, total_var)

        left = 0.0                               # running start of the next segment
        for lev in LEVELS:
            width = tau2[lev]                    # segment length = tau_k^2 itself
            share = width / total_var if total_var > 0 else 0.0
            # The label goes only on the first bar, so the legend gets one
            # entry per level rather than one per (model, level).
            kw = st.kw(lev, "bar", {"facecolor": LEVEL_COLOURS[lev], "edgecolor": "w",
                                    "linewidth": 0.8, "height": 0.6,
                                    "label": lev if y == 0 else None})
            if kw is not None:
                if y != 0:
                    kw["label"] = None
                ax.barh(y, width, left=left, **kw)
            if show_labels and share >= share_threshold:
                kw = st.kw("segment_text", "text", {"ha": "center", "va": "center",
                                                    "fontsize": 7, "color": "w"})
                if kw is not None:
                    ax.text(left + width / 2, y, f"{share:.0%}", **kw)
            left += width

        if show_labels:
            # Total variance at the end of the bar. The SD is kept alongside
            # because exp(tau_tot) is the multiplicative spread engineers read.
            kw = st.kw("total_text", "text", {"ha": "left", "va": "center", "fontsize": 8})
            if kw is not None:
                ax.text(total_var, y,
                        rf"  {total_var:.4f}  ($\tau_{{tot}}$={np.sqrt(total_var):.3f})", **kw)

    ax.set_yticks(y_pos, list(labels) if labels is not None else [vw.model_id for vw in views])
    ax.invert_yaxis()                            # first model in the list on top
    ax.set_xlim(0, x_max * 1.45 if x_max > 0 else 1.0)   # room for the end-of-bar text
    ax.set_xlabel(rf"between-row variance of $\ln(\{quantity}_{{MSA}}/\{quantity}_{{IDA}})$, "
                  r"$\tau_d^2+\tau_s^2+\tau_u^2$")
    ax.set_title(rf"Variance decomposition: $\{quantity}$", loc="left")
    # Legend above the axes, top right, in one row, so it never sits on a bar
    # or on the end-of-bar text however many models are plotted.
    kw = st.kw("legend", "dict", {"fontsize": 8, "ncol": 3, "frameon": False,
                                  "loc": "lower right", "bbox_to_anchor": (1.0, 1.0)})
    if kw is not None:
        ax.legend(**kw)
    kw = st.kw("grid", "dict", {"axis": "x", "alpha": 0.3})
    if kw is not None:
        ax.grid(**kw)
    ax.set_axisbelow(True)
    return ax


# =============================================================================
# Fitted-value forest plot
# =============================================================================


def plot_fitted_forest(ax, models, quantity: str | None, sort_covariate: str,
                       significance_level: float = 0.05, add_sampling: bool = False,
                       label_fontsize: float = 5, *, style: Mapping | None = None,
                       model_styles=None, colour_by: str = "auto"):
    """Fitted-value forest plot: one horizontal line of the plot per row (site, structure).

    For every row i:
      - a SQUARE at the observed ratio exp(y_i), with its (1 - alpha) sampling
        interval from the record bootstrap ("what the data say");
      - per model, a DIAMOND at the fitted ratio exp(x_i' beta_hat), with a
        (1 - alpha) PREDICTION interval ("what the model says a structure with
        these covariates should look like").

    Sorting the rows by ``sort_covariate`` makes the diamonds trace out the
    regression line turned on its side (when the covariate is in the model).
    Squares outside their diamond's interval are rows the model does not explain
    and are ringed. This is metafor's ``addfit`` forest plot.

    The prediction interval on the log scale is

        y_fit_i +/- z * sqrt( x_i' Var(beta_hat) x_i + tau_tot^2 [+ v_i] )

    (Borenstein et al. [1], Ch. 17, ~pp. 127-133, approximate). The observed
    interval is the percentile interval of the bootstrap replicates, which
    exponentiates straight onto the ratio scale (Efron & Tibshirani [5], Ch. 13,
    ~pp. 170-174, approximate).

    With several models, the observed square is drawn once (from the first
    model) and each model gets its own diamond, stacked below it within the row.
    The models must share the same rows; if their observed values differ (e.g.
    one bias-corrected and one not) a warning is issued and the first is shown.

    Parameters
    ----------
    ax : matplotlib Axes
    models : fit or list of fits
    quantity : {"theta", "beta"} or None
    sort_covariate : str
        Column of the contrast rows (in the model or not), or an index level
        ("site", "n_storeys"). Rows are ordered by it, smallest at the top.
    significance_level : float
        alpha; both intervals are (1 - alpha).
    add_sampling : bool
        False: the PI is for the TRUE effect of a new structure with these
        covariates. True: also add the row's own sampling variance v_i, so the
        interval is where the OBSERVED y_i should land -- the fair yardstick for
        "is this square an outlier?".
    label_fontsize : float
        Font size of the per-row y tick labels.
    style, model_styles, colour_by
        See the module docstring. ``colour_by`` affects the diamonds; the
        observed squares are always coloured by storey.

    Elements
    --------
    pi_interval    ax.hlines   fitted prediction interval, per model per storey
    fitted_points  ax.scatter  fitted diamonds, per model per storey
    obs_interval   ax.hlines   observed sampling interval, per storey
    obs_points     ax.scatter  observed squares, per storey
    outside        ax.scatter  rings round observed points outside a model's PI
    vline          ax.axvline  x = 1
    grid           ax.grid     keywords for the grid (None hides it)

    Returns
    -------
    ax
    """
    import warnings

    views, _ = _as_views(models, quantity)
    st = _Styler(style, model_styles, views, colour_by)
    q = views[0].quantity
    ref = views[0]
    n = ref.n
    z_crit = stats.norm.ppf(1 - significance_level / 2)
    lev = _level(significance_level)
    multi = st.K > 1

    for vw in views[1:]:
        if not vw.index.equals(ref.index):
            raise ValueError(f"{vw.model_id} and {ref.model_id} do not have the same rows")
        if not np.allclose(vw.y_obs, ref.y_obs):
            warnings.warn(f"{vw.model_id} and {ref.model_id} have different observed values; "
                          f"the observed squares are those of {ref.model_id}", stacklevel=2)

    # --- observed values and their sampling intervals ------------------------
    y_obs = ref.y_obs
    q_obs = ref.contrast.quantiles((significance_level / 2,
                                    1 - significance_level / 2)).to_numpy(float)
    obs_lo, obs_hi = q_obs[:, 0], q_obs[:, 1]

    # --- sort order -----------------------------------------------------------
    s_val = ref.covariate_values(sort_covariate)
    # Stable sort: ties (e.g. the two structures at a site sharing a site
    # covariate) keep their original order.
    order = np.argsort(s_val, kind="stable")
    y_pos = np.empty(n)
    y_pos[order] = np.arange(n)[::-1]           # first sorted row at the top

    # Vertical slots within each row: the observed square on top, then one per
    # model. One model reproduces the original +/- 0.2.
    spread = 0.2 if st.K == 1 else 0.35
    slots = np.linspace(spread, -spread, st.K + 1)
    off_obs, off_fit = slots[0], slots[1:]

    # --- per model: fitted value and prediction interval -----------------------
    for k, vw in enumerate(views):
        y_fit = vw.fitted
        var_pi = vw.var_fitted + vw.tau2_total          # + tau_tot^2 -> new site/design
        if add_sampling:
            var_pi = var_pi + vw.v                      # + v_i -> where observed y_i lands
        pi_lo = y_fit - z_crit * np.sqrt(var_pi)
        pi_hi = y_fit + z_crit * np.sqrt(var_pi)
        yk = y_pos + off_fit[k]                          # off_fit < 0: below the square
        pre = "{model}: " if multi else ""

        for ns, sel in _storey_groups(vw):
            kw = st.kw("pi_interval", "hlines",
                       {"color": st.series_colour(k, "0.55"), "linewidth": 1.6,
                        "alpha": 0.6, "zorder": 1}, k=k, ns=ns)
            if kw is not None:
                ax.hlines(yk[sel], np.exp(pi_lo[sel]), np.exp(pi_hi[sel]), **kw)
            kw = st.kw("fitted_points", "scatter",
                       {"marker": _fitted_marker(st, k),
                        "s": 9, "facecolor": "w", "edgecolor": st.point_colour(k, ns),
                        "linewidth": 0.9, "zorder": 3,
                        "label": pre + f"{_storey_text(ns)} fitted, {{level}} PI"},
                       k=k, ns=ns, fmt={"level": lev})
            if kw is not None:
                ax.scatter(np.exp(y_fit[sel]), yk[sel], **kw)

        # Observed points outside this model's PI. Expect about alpha * n if
        # the model is right (only exactly so with add_sampling=True).
        outside = (y_obs < pi_lo) | (y_obs > pi_hi)
        kw = st.kw("outside", "scatter",
                   {"marker": "o", "s": 49 + 40 * k, "facecolor": "none",
                    "edgecolor": st.series_colour(k, "k"), "linewidth": 0.8, "zorder": 5,
                    "label": pre + "observed outside PI ({n_out} of {n})"},
                   k=k, fmt={"n_out": int(outside.sum()), "n": n})
        if kw is not None:
            ax.scatter(np.exp(y_obs[outside]), y_pos[outside] + off_obs, **kw)

    # --- observed squares, drawn once ----------------------------------------
    for ns, sel in _storey_groups(ref):
        col = st.storey_colour(ns) if ns is not None else "k"
        kw = st.kw("obs_interval", "hlines", {"color": col, "linewidth": 0.8, "alpha": 0.7,
                                              "zorder": 2}, ns=ns)
        if kw is not None:
            ax.hlines(y_pos[sel] + off_obs, np.exp(obs_lo[sel]), np.exp(obs_hi[sel]), **kw)
        kw = st.kw("obs_points", "scatter",
                   {"marker": "s", "s": 9, "facecolor": col, "edgecolor": "k",
                    "linewidth": 0.4, "zorder": 4,
                    "label": f"{_storey_text(ns)} observed, {{level}} CI"},
                   ns=ns, fmt={"level": lev})
        if kw is not None:
            ax.scatter(np.exp(y_obs[sel]), y_pos[sel] + off_obs, **kw)

    kw = st.kw("vline", "line", {"color": "0.4", "linewidth": 0.9, "linestyle": "--",
                                 "zorder": 0})
    if kw is not None:
        ax.axvline(1.0, **kw)                    # no difference SS vs FX

    # --- labels -----------------------------------------------------------------
    # "site Ns  (sort value)" per row, so the ordering can be read off.
    idx = ref.index
    numeric = np.issubdtype(np.asarray(s_val).dtype, np.number)
    if "site" in (idx.names or []):
        sites = idx.get_level_values("site")
        tick = [f"{s} {ns}s" for s, ns in zip(sites, ref.storeys)]
    else:
        tick = list(ref.contrast.labels)
    if numeric:
        tick = [f"{t}  ({v:.3g})" for t, v in zip(tick, s_val)]
    ax.set_yticks(y_pos, tick, fontsize=label_fontsize)
    ax.set_ylim(-1, n)
    ax.set_xlabel(ratio_label(q))
    ax.set_ylabel(f"row, sorted by {sort_covariate}")
    title = (f"{ref.model_id}: observed vs fitted ({q})" if not multi
             else f"{', '.join(v.model_id for v in views)}: observed vs fitted ({q})")
    ax.set_title(title, loc="left")
    kw = st.kw("grid", "dict", {"axis": "x", "linestyle": "-.", "color": "0.8"})
    if kw is not None:
        ax.grid(**kw)
    ax.set_axisbelow(True)
    return ax


# =============================================================================
# Whitened residuals
# =============================================================================


#: Lines the text box of :func:`plot_whitened_residuals` can hold, in the
#: default order (all but "model").
STATS_ITEMS: tuple[str, ...] = ("model", "method", "Q", "r_lims", "Shapiro-Wilk")
_DEFAULT_STATS_ITEMS: tuple[str, ...] = ("method", "Q", "r_lims", "Shapiro-Wilk")
# Matching key -> canonical name. The key ignores case, "-", "_" and spaces.
_STATS_ALIASES: dict[str, str] = {
    **{name.lower().replace("-", "").replace("_", ""): name for name in STATS_ITEMS},
    "sw": "Shapiro-Wilk", "shapiro": "Shapiro-Wilk",
}


def _stats_items(items: Sequence[str] | None) -> list[str]:
    """Canonical names of the requested text-box lines, in the order requested."""
    if items is None:
        return list(_DEFAULT_STATS_ITEMS)
    if isinstance(items, str):          # a single name, not its characters
        items = [items]
    out = []
    for it in items:
        key = str(it).lower().replace("-", "").replace("_", "").replace(" ", "")
        if key not in _STATS_ALIASES:
            raise ValueError(f"unknown stats item {it!r}; choose from {list(STATS_ITEMS)}")
        name = _STATS_ALIASES[key]
        if name not in out:
            out.append(name)
    return out


def whitened_residuals(model, quantity: str | None = None, method: str = "symmetric"
                       ) -> pd.DataFrame:
    """Residuals of one outcome, with the row-to-row correlation and unequal precision removed.

    The plain residual ``e_i = y_i - x_i' beta_hat`` is a poor diagnostic here,
    because (i) rows differ in how noisy they are (v_i and the tau's), and (ii)
    rows are correlated: two structures at a site share a site effect and
    records, and every row shares the fixed record set (dense V_bb). A run of
    positive residuals can simply be the shared record set, not a model failure.

    Whitening fixes both. With Sigma the total covariance the fit used
    (:attr:`FitView.Sigma`), take a matrix square root ``Sigma = R R'`` and form

        r* = R^{-1} e.

    If the model is right, ``r* ~ N(0, I)`` -- approximately: estimating beta
    costs p degrees of freedom, so each ``Var(r*_i)`` is slightly below 1 and
    ``sum r*^2 ~ chi^2_{n-p}`` (Gelman et al. BDA3 [2], Ch. 14.7, ~pp. 369-371,
    approximate, for the whitening of a GLS regression).

    Parameters
    ----------
    model : ModelFit, JointModelFit or FitView
    quantity : {"theta", "beta"} or None
        For a joint fit, the outcome whose MARGINAL covariance (its diagonal
        block of Sigma) is used; the theta-beta cross terms are not needed to
        whiten one outcome.
    method : {"symmetric", "cholesky"}
        Which square root R to use. Both give ``r* ~ N(0, I)``; they differ in
        how ``r*_i`` relates to row i.
        "symmetric": ``R = Sigma^{1/2}`` (eigen-decomposition). The whitened
            vector closest to the plain standardised residuals, so ``r*_i``
            still "belongs" to row i and can be plotted against that row's
            covariates. Default.
        "cholesky": ``R = L``, lower triangular. ``r*_i`` is row i after
            removing what the rows BEFORE it (in index order) predict about it.
            Its meaning depends on the row order, so use it for the global
            checks (QQ plot, Q statistic), not for reading individual rows.

    Returns
    -------
    DataFrame indexed like the rows, columns:
        fitted     x_i' beta_hat (log scale)
        resid      e_i = y_i - fitted (log scale)
        r_std      e_i / sqrt(Sigma_ii): scaled for precision but NOT de-correlated
        r_white    r*_i
    """
    if method not in ("symmetric", "cholesky"):
        raise ValueError(f"method must be 'symmetric' or 'cholesky', got {method!r}")
    vw = fit_view(model, quantity)
    fitted = vw.fitted
    e = vw.y_obs - fitted
    S = vw.Sigma

    if method == "cholesky":
        L = np.linalg.cholesky(S)                          # S = L L'
        r_white = solve_triangular(L, e, lower=True)       # L^{-1} e, no explicit inverse
    else:
        # S = U diag(lam) U'  ->  S^{-1/2} = U diag(lam^{-1/2}) U'
        lam, U = np.linalg.eigh(S)
        r_white = U @ ((U.T @ e) / np.sqrt(lam))

    return pd.DataFrame({"fitted": fitted, "resid": e,
                         "r_std": e / np.sqrt(np.diag(S)),
                         "r_white": r_white}, index=vw.index)


def _draw_r_vs_x(ax, st: _Styler, k: int, vw: FitView, x: np.ndarray, r: np.ndarray,
                 w: int, pre: str) -> None:
    """Draw model k's whitened residuals ``r`` against ``x`` on one axes.

    Shared by the "r* vs fitted" and "r* vs covariate" panels of
    :func:`plot_whitened_residuals` and by every panel of
    :func:`plot_whitened_residuals_vs_covariates`, so they all look the same.

    Two elements, in this order:
      "points"        one scatter per storey count, so the 3- and 5-storey rows
                      can be told apart (colour or marker, see ``colour_by``);
      "running_mean"  a centred rolling mean of r* taken in order of x. A single
                      outlier barely moves it, but a systematic slope or bend
                      does, which is what the panel is for.

    Parameters
    ----------
    ax : Axes
    st : _Styler
        Resolves each element's keywords (and hides it if styled None).
    k : int
        Position of the model in the overlaid list (picks its colour/marker).
    vw : FitView
        The model's view; only used to split the rows by storey.
    x, r : ndarray (n,)
        x values and whitened residuals, aligned with ``vw``'s rows.
    w : int
        Window of the running mean, in rows. ``min_periods = w // 2`` lets the
        mean run to the ends of the axis on half a window instead of stopping
        w/2 rows short.
    pre : str
        Label prefix: ``"{model}: "`` when several models share the axes, so the
        legend entries say which model they belong to; "" otherwise.
    """
    for ns, sel in _storey_groups(vw):
        kw = st.kw("points", "scatter",
                   {"marker": st.point_marker(k, ns), "s": 14,
                    "facecolor": st.point_colour(k, ns), "edgecolor": "k",
                    "linewidth": 0.3, "alpha": 0.7,
                    "label": pre + _storey_text(ns)},
                   k=k, ns=ns)
        if kw is not None:
            ax.scatter(x[sel], r[sel], **kw)
    # Running mean (centred window) to make a trend or a curve visible. Sorting
    # by x first means "neighbours" are rows with similar x, not adjacent rows.
    o = np.argsort(x)
    kw = st.kw("running_mean", "line",
               {"color": st.series_colour(k, "k"), "linewidth": 1.2,
                "label": pre + f"running mean ({w} rows)"}, k=k)
    if kw is not None:
        ax.plot(x[o], pd.Series(r[o]).rolling(w, center=True,
                                              min_periods=w // 2).mean(), **kw)


def _residual_axes_furniture(ax, st: _Styler, z_crit: float, label_crit: bool) -> None:
    """Reference lines and grid of one whitened-residual axes.

    Elements, in this order:
      "zero_line"   r* = 0, where the residuals should centre;
      "crit_lines"  r* = +/- z_crit, the two-sided (1 - alpha) limit for a SINGLE
                    standard-normal residual;
      "grid"        keywords for ``ax.grid`` (None: no grid).
    The grid is put behind the data with ``set_axisbelow``.

    ``label_crit`` attaches the crit-line label to the lower line only, so a
    legend drawn on this axes gets ONE entry for the pair. Pass True on exactly
    one axes per figure (the one carrying the legend), otherwise the label would
    be repeated in a figure-level legend.
    """
    kw = st.kw("zero_line", "line", {"color": "0.4", "linewidth": 0.8})
    if kw is not None:
        ax.axhline(0, **kw)
    kw = st.kw("crit_lines", "line", {"color": "0.4", "linewidth": 0.8, "linestyle": "--",
                                      "label": r"$\pm${z:.2f} limit (single $r^*$)"},
               fmt={"z": z_crit})
    if kw is not None:
        for j, s in enumerate((-z_crit, z_crit)):
            lab = kw.get("label") if (label_crit and j == 0) else None
            ax.axhline(s, **{**kw, "label": lab})
    kw = st.kw("grid", "dict", {"linestyle": "-.", "color": "0.8"})
    if kw is not None:
        ax.grid(**kw)
    ax.set_axisbelow(True)


def plot_whitened_residuals(axes, models, quantity: str | None, sort_covariate: str,
                            method: str = "symmetric", significance_level: float = 0.05,
                            *, running_window: int | None = None, show_stats: bool = True,
                            stats_items: Sequence[str] | None = None, qq_legend: bool = True,
                            style: Mapping | None = None, model_styles=None,
                            colour_by: str = "auto"):
    """Three diagnostic panels of the whitened residuals r* for one outcome.

    axes[0]  Normal QQ plot with a pointwise (1 - alpha) envelope.
             Points on the line -> the normal assumption for the true effects
             holds. An S-shape bending away at both ends -> heavier tails than
             normal (a few rows much further out than tau allows).
    axes[1]  r* against the fitted ratio.
             A funnel -> spread changes with the level of the effect.
             A curve -> the linear-in-covariates form is wrong somewhere.
    axes[2]  r* against ``sort_covariate`` (any column of the rows).
             A trend for a covariate NOT in the model -> it would explain
             something. A curve for one IN the model -> it enters non-linearly.

    The text box on axes[0] holds the global checks chosen by ``stats_items``
    (see :data:`STATS_ITEMS`), in the order given:
        "model"         the model id
        "method"        the whitening method
        "Q"             Q = sum r*^2 against chi^2 on (n - p) degrees of freedom
                        -- the generalised Cochran Q (Borenstein et al. [1],
                        Ch. 16, ~pp. 107-111, approximate, for the univariate
                        form); Q >> n - p means more scatter than the model allows.
        "r_lims"        the count of |r*| > z_crit (outside the dashed lines),
                        against the alpha * n expected (a fair test, because the
                        r* are independent).
        "Shapiro-Wilk"  the Shapiro-Wilk p-value for normality.

    Two different "limits" appear on the QQ plot, answering different questions:
      * the DASHED horizontal lines at +/- z_crit (1.96 for alpha = 0.05) ask
        "is this one residual unusually large?". They are what "r_lims" counts.
      * the GREY envelope asks "is the i-th smallest of the n residuals where the
        i-th smallest of n standard normals should be?". It uses the fact that
        the i-th of n sorted uniforms follows Beta(i, n + 1 - i); mapping its
        quantiles through the normal ppf gives a pointwise band for the i-th
        sorted normal. It is narrow in the middle (middle order statistics of
        many draws are tightly pinned) and wide at the ends, so a point on its
        edge near the centre is slightly off its expected RANK, not a large
        residual. Being pointwise, a few of n points outside it are expected even
        when the model is right; a systematic run outside it is the warning.

    Parameters
    ----------
    axes : sequence of 3 Axes
    models : fit or list of fits
        Several models are overlaid on each panel; the envelope is drawn once.
    quantity : {"theta", "beta"} or None
    sort_covariate : str
        Column of the contrast rows, or an index level.
    method : {"symmetric", "cholesky"}
        See :func:`whitened_residuals`.
    significance_level : float
    running_window : int, optional
        Window (rows) of the running mean on panels 1-2; default ``max(5, n // 10)``.
    show_stats : bool
        Draw the text box of global checks on axes[0].
    stats_items : sequence of str, optional
        Which lines the text box holds, and in what order: any of "model",
        "method", "Q", "r_lims", "Shapiro-Wilk" (matching ignores case, "-", "_"
        and spaces, so "shapiro_wilk" and "rlims" also work; "SW" is short for
        Shapiro-Wilk). Default ``("method", "Q", "r_lims", "Shapiro-Wilk")``.
        With several models, "method" (common to all) is written once at the top
        and each model's block is headed by its id whether or not "model" is
        listed. An empty sequence draws no box.
    qq_legend : bool
        Draw a legend on the QQ panel naming the envelope, the +/- z_crit lines
        and the points (keywords via the "qq_legend" element).
    style, model_styles, colour_by
        See the module docstring.

    Elements
    --------
    envelope      ax.fill_between  QQ envelope (axes[0])
    qq_line       ax.plot          y = x on the QQ plot
    qq_points     ax.scatter       QQ points, per model per storey
    points        ax.scatter       r* on axes[1] and axes[2], per model per storey
    running_mean  ax.plot          running mean on axes[1] and axes[2], per model
    zero_line     ax.axhline       r* = 0 on every panel
    crit_lines    ax.axhline       r* = +/- z_crit on every panel
    stats_text    ax.text          the global-check box ("bbox" for its frame)
    qq_legend     ax.legend        keywords for the QQ-panel legend (None hides it)
    grid          ax.grid          keywords for the grid (None hides it)

    Returns
    -------
    DataFrame of :func:`whitened_residuals` for a single model, or
    ``{model_id: DataFrame}`` when a list of models was passed.
    """
    views, single = _as_views(models, quantity)
    st = _Styler(style, model_styles, views, colour_by)
    q = views[0].quantity
    z_crit = stats.norm.ppf(1 - significance_level / 2)
    multi = st.K > 1

    # --- QQ envelope, from the first model's n --------------------------------
    n0 = views[0].n
    i = np.arange(1, n0 + 1)
    theo0 = stats.norm.ppf((i - 0.5) / n0)                 # plotting position (i - 0.5)/n
    env_lo = stats.norm.ppf(stats.beta.ppf(significance_level / 2, i, n0 + 1 - i))
    env_hi = stats.norm.ppf(stats.beta.ppf(1 - significance_level / 2, i, n0 + 1 - i))
    kw = st.kw("envelope", "fill", {"facecolor": "0.85", "linewidth": 0,
                                    "label": "{level} envelope of sorted $r^*$"},
               fmt={"level": _level(significance_level)})
    if kw is not None:
        axes[0].fill_between(theo0, env_lo, env_hi, **kw)
    kw = st.kw("qq_line", "line", {"color": "k", "linewidth": 0.9})
    if kw is not None:
        axes[0].plot(theo0, theo0, **kw)

    results, stat_lines = {}, []
    for k, vw in enumerate(views):
        res = whitened_residuals(vw, method=method)
        results[vw.model_id] = res
        r = res["r_white"].to_numpy()
        n = r.size
        pre = "{model}: " if multi else ""

        # (a) QQ: sort the residuals; theo[j] pairs with the j-th smallest r.
        theo = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        srt = np.argsort(r)
        r_sorted, storeys_sorted = r[srt], vw.storeys[srt]
        for ns, _ in _storey_groups(vw):
            sel = np.array([s == ns for s in storeys_sorted])
            kw = st.kw("qq_points", "scatter",
                       {"marker": st.point_marker(k, ns), "s": 12,
                        "facecolor": st.point_colour(k, ns), "edgecolor": "k",
                        "linewidth": 0.3, "label": pre + _storey_text(ns)},
                       k=k, ns=ns)
            if kw is not None:
                axes[0].scatter(theo[sel], r_sorted[sel], **kw)

        # (b) r* vs fitted ratio, (c) r* vs sort_covariate
        x_c = vw.covariate_values(sort_covariate).astype(float)
        w = running_window or max(5, n // 10)
        for ax, x in ((axes[1], np.exp(res["fitted"].to_numpy())), (axes[2], x_c)):
            _draw_r_vs_x(ax, st, k, vw, x, r, w, pre)

        # Global checks: one line per item, keyed by its STATS_ITEMS name.
        p_q = len(vw.x_names)                            # coefficients for this outcome
        Q = float(r @ r)
        df = n - p_q
        p_Q = stats.chi2.sf(Q, df)
        n_out = int((np.abs(r) > z_crit).sum())
        p_sw = stats.shapiro(r).pvalue
        stat_lines.append({
            "model": vw.model_id,
            "method": f"method: {method}",
            "Q": f"Q = {Q:.1f} on {df} df (p = {p_Q:.2f})",
            "r_lims": f"|r*| > {z_crit:.2f}: {n_out} (expect {significance_level * n:.1f})",
            "Shapiro-Wilk": f"Shapiro-Wilk p = {p_sw:.2f}",
        })

    axes[0].set_xlabel("standard normal quantile")
    axes[0].set_ylabel(r"whitened residual $r^*$")
    axes[0].set_title(f"({q}) normal QQ", loc="left")
    axes[1].set_xlabel(rf"fitted $\{q}_{{SS}}/\{q}_{{FX}}$")
    axes[2].set_xlabel(sort_covariate)
    axes[1].set_title(f"({q}) vs fitted", loc="left")
    axes[2].set_title(f"({q}) vs {sort_covariate}", loc="left")

    for ax in axes:
        # The crit-line label goes on the QQ panel only, where the legend is.
        _residual_axes_furniture(ax, st, z_crit, label_crit=ax is axes[0])

    items = _stats_items(stats_items)
    if show_stats and items:
        if multi:
            # "method" is common to every model: once, at the top. Each model's
            # block is headed by its id so the numbers can be told apart.
            top = [stat_lines[0]["method"]] if "method" in items else []
            blocks = ["\n".join([d["model"]] + [d[it] for it in items
                                                if it not in ("model", "method")])
                      for d in stat_lines]
            text = "\n".join(top + blocks)
        else:
            text = "\n".join(stat_lines[0][it] for it in items)
        kw = st.kw("stats_text", "text", {"ha": "left", "va": "top", "fontsize": 8,
                                          "bbox": dict(facecolor="w", edgecolor="0.7")})
        if kw is not None:
            axes[0].text(0.03, 0.97, text, transform=axes[0].transAxes, **kw)

    if qq_legend:
        kw = st.kw("qq_legend", "dict", {"fontsize": 7, "loc": "lower right"})
        if kw is not None:
            axes[0].legend(**kw)

    return results[views[0].model_id] if single else results


# =============================================================================
# Whitened residuals against a list of covariates
# =============================================================================

#: Default look of the "in model" note above each panel of
#: :func:`plot_whitened_residuals_vs_covariates`, per status:
#:   "in"     every overlaid model includes the covariate for this outcome;
#:   "out"    none does -- emphasised, because a trend in r* against a covariate
#:            the model leaves out is the finding this plot exists to show;
#:   "mixed"  some models include it and some do not; the text names which.
#: ``"s"`` is the note's text, a template with the fields ``{in_models}``,
#: ``{out_models}`` (comma-separated model ids) and ``{covariate}``. Every other
#: key is an ``ax.text`` keyword. Override per call with
#: ``style["model_note_status"]`` rather than by editing this dict.
MODEL_NOTE_STYLE: dict[str, dict] = {
    "in": {"s": "in model", "color": "k"},
    "out": {"s": "not in model", "color": "#ff0000", "fontweight": "bold"},
    "mixed": {"s": "in: {in_models} · not in: {out_models}", "color": "k"},
}

#: Covariates that are properties of the DESIGN (constant across the sites a
#: design group spans), so they keep their meaning in a group-scope contrast
#: (A2, one row per design group). Taken from the design factors nb 072 writes.
#: Any other column is treated as site-level: at group scope the contrast rows
#: are ``groupby("design_group_id").first()``, so such a column holds the value
#: of the group's FIRST site, and plotting against it is not meaningful.
#: A ``_corrected`` suffix is ignored when looking a name up here.
DESIGN_LEVEL_COVARIATES: tuple[str, ...] = ("ln_T1", "ln_overstrength", "ln_ductility",
                                            "n_storeys")


def _available_covariates(vw: FitView) -> list[str]:
    """Names :meth:`FitView.covariate_values` accepts, for error messages.

    The factor-table columns minus the per-arm columns (the same filter
    :func:`fm.build_X` uses in its own error message), then the index levels.
    """
    cols = [c for c in vw.contrast.rows.columns
            if not any(str(c).startswith(p + "_") for p in fm.ARMS)]
    return cols + [nm for nm in (vw.index.names or []) if nm is not None]


def _check_covariates(views: Sequence[FitView], covs: Sequence[str]) -> None:
    """Raise before anything is drawn if any covariate is unknown or not numeric.

    Every (model, covariate) pair is checked and all problems are reported at
    once, so a typo in the fifth name does not leave four panels drawn and the
    fifth missing.
    """
    unknown: dict[str, list[str]] = {}          # model_id -> unknown names
    for vw in views:
        for cov in covs:
            try:
                vals = vw.covariate_values(cov)
            except KeyError:
                unknown.setdefault(vw.model_id, []).append(cov)
                continue
            # A text column ("design_group_id", "tag") or object-typed site labels
            # cannot go on a numeric axis or into a running mean.
            if not pd.api.types.is_numeric_dtype(np.asarray(vals).dtype):
                raise ValueError(
                    f"covariate {cov!r} of {vw.model_id} ({vw.quantity}) is not numeric "
                    f"(dtype {np.asarray(vals).dtype}); it cannot be an x axis")
    if unknown:
        vw = next(v for v in views if v.model_id in unknown)
        lines = "; ".join(f"{mid}: {names}" for mid, names in unknown.items())
        raise KeyError(f"unknown covariate(s) - {lines}. Available for {vw.model_id} "
                       f"({vw.quantity}): {_available_covariates(vw)}")


def _warn_group_scope(views: Sequence[FitView], covs: Sequence[str]) -> None:
    """Warn for a site-level covariate plotted against a group-scope outcome.

    Only for covariates NOT in that model: one the model does include is a
    property of the model itself, not something this plot introduces.
    """
    for vw in views:
        if vw.contrast.scope != "group":
            continue
        for cov in dict.fromkeys(covs):                 # each name once, in order
            base = cov[:-len("_corrected")] if cov.endswith("_corrected") else cov
            if cov in vw.x_names or base in DESIGN_LEVEL_COVARIATES:
                continue
            warnings.warn(
                f"{vw.model_id} ({vw.quantity}) is a group-scope contrast (one row per "
                f"design group): {cov!r} is a site-level column, so each group shows the "
                f"value of its FIRST site, not a property of the design. A trend against "
                f"it is not interpretable.", UserWarning, stacklevel=3)


def _panel_letter(j: int) -> str:
    """'(A)', '(B)', ... for panel j; numbers after Z."""
    return f"({chr(65 + j)})" if j < 26 else f"({j + 1})"


def _model_note_kw(style: Mapping, status: str, cov: str) -> dict | None:
    """Final ``ax.text`` keywords (with the text under "s") of one panel's note.

    Layers, later winning; a layer that is None hides the note:
        built-in placement and size
        MODEL_NOTE_STYLE[status]
        style["model_note_status"][status]   every note of this status
        style["model_note"]                  every note
        style["model_note_covariates"][cov]  this panel only
    The last layer is the most specific, so it can rewrite any one note by hand.
    ``model_styles`` does not reach the note: the note belongs to the panel, not
    to one of the overlaid models.
    """
    by_status = style.get("model_note_status", {}) or {}
    by_cov = style.get("model_note_covariates", {}) or {}
    layers = [
        {"fontsize": 8, "ha": "left", "va": "bottom", "clip_on": False},
        MODEL_NOTE_STYLE[status],
        by_status.get(status, {}),
        style.get("model_note", {}),
        by_cov.get(cov, {}),
    ]
    out: dict = {}
    for layer in layers:
        if layer is None:
            return None
        if layer:
            out.update(_normalise("text", layer))
    return out


def _is_first_column(ax, j: int, ncols: int) -> bool:
    """True if ``ax`` sits in the first column of its grid.

    Read from the axes' own subplot spec, so it is also right for axes the
    caller laid out; an axes with no spec (placed by hand) counts as first.
    """
    spec = ax.get_subplotspec() if hasattr(ax, "get_subplotspec") else None
    if spec is not None:
        return spec.is_first_col()
    return j % ncols == 0


def plot_whitened_residuals_vs_covariates(
        models, quantity: str | None, covariates: Sequence[str] | str, *,
        axes=None, ncols: int = 3, method: str = "symmetric",
        significance_level: float = 0.05, running_window: int | None = None,
        sharey: bool = True, covariate_labels: Mapping[str, str] | None = None,
        model_note: bool = True, panel_letters: bool = True, legend: bool = True,
        style: Mapping | None = None, model_styles=None, colour_by: str = "auto",
        figsize_per_panel: tuple[float, float] = (5.0, 4.0),
        save_path=None, dpi: int = 150):
    """Whitened residuals r* of one outcome against each of several covariates.

    One panel per covariate, each drawn exactly like the third panel of
    :func:`plot_whitened_residuals`: r*_i against row i's covariate value, the
    rows split by storey, a running mean, the r* = 0 line and the +/- z_crit
    lines. Above each panel a note says whether that covariate is IN THE MODEL
    for this outcome.

    How to read it
    --------------
    r* is the residual with the rows' unequal precision and their correlation
    (shared site effect, shared fixed record set) removed, ``r* = Sigma^{-1/2} e``
    (Gelman et al. BDA3 [2], Ch. 14.7, ~pp. 369-371, approximate, for whitening
    a GLS residual; see :func:`whitened_residuals`). If the model is right the
    r* are roughly independent N(0, 1), so each panel should be a flat,
    structureless cloud around 0, mostly inside the dashed lines. A pattern
    against a predictor points at a missing or mis-specified term (Chatterjee &
    Hadi [7], Ch. 4, ~pp. 97-107, approximate, residuals plotted against each
    predictor; sec. 4.12, ~pp. 108-112, covers the related partial-residual plot
    of :func:`plot_conditional_effect`):

    * a TREND against a covariate NOT in the model -> it would explain
      something the model leaves in the residuals;
    * a CURVE against one IN the model -> it enters, but not linearly.

    The note is there because the two cases call for different readings of the
    same picture. By default "not in model" is bold red, so the panels that can
    reveal a missing term stand out.

    The x axis is the covariate AS STORED -- a log for the ``ln_*`` and
    ``d_ln_*`` columns -- the same axis panel (c) of
    :func:`plot_whitened_residuals` uses. That is why the default x label is
    the column name and not :data:`DEFAULT_COVARIATE_LABELS`, whose wording
    describes the ratio (exp'd) scale.

    Which covariates are "in the model" is decided per model and per outcome
    (``covariate in view.x_names``), so for a joint fit theta and beta can
    differ: e.g. A8f has five covariates for theta but only
    ``d_ln_ds595_sigma_unif`` for beta. When several overlaid models disagree
    on a covariate, the note names them: "in: A8f_bc_hb · not in: A8e_bc_hb".

    Parameters
    ----------
    models : fit or list of fits
        ModelFit, JointModelFit or FitView; several are overlaid on every panel.
        Models with different numbers of rows can be overlaid; each uses its own.
    quantity : {"theta", "beta"} or None
    covariates : str or sequence of str
        Columns of the contrast rows, or index levels (e.g. "n_storeys"), one
        panel each in the order given. A single string means ONE covariate.
    axes : array-like of Axes, optional
        Draw into these (flattened, row-major) instead of a new figure; needs at
        least ``len(covariates)``. The caller then owns the layout: no suptitle,
        no ``tight_layout`` and spare axes are left as they are.
    ncols : int
        Panels per row of a new figure.
    method : {"symmetric", "cholesky"}
        See :func:`whitened_residuals`. Keep "symmetric" here: Cholesky r*_i
        depend on the row order, so they do not belong to row i's covariates.
    significance_level : float
        alpha of the +/- z_crit lines, ``z_crit = Phi^{-1}(1 - alpha / 2)``.
    running_window : int, optional
        Window of the running mean, in rows; default ``max(5, n // 10)`` per model.
    sharey : bool
        Share the y axis across the panels of a new figure, with the y label on
        the first column only. False labels every panel.
    covariate_labels : {covariate: label}, optional
        x labels; a covariate not listed is labelled with its column name.
    model_note : bool
        Draw the in/out note. False is the same as ``style={"model_note": None}``.
    panel_letters : bool
        Prefix each note with "(A)", "(B)", ... . When the note is not drawn the
        letter alone becomes the panel's left title.
    legend : bool
        One legend, on the first panel (keywords via the "legend" element).
    style, model_styles, colour_by
        See the module docstring. The note is styled through ``style`` only (it
        belongs to the panel, not to a model), see "Elements".
    figsize_per_panel : (width, height)
        Size of one panel of a new figure, in inches.
    save_path : path, optional
        Save the figure there, ``facecolor="w"``.
    dpi : int
        Resolution of the saved figure.

    Elements
    --------
    points        ax.scatter  r*, per model per storey
    running_mean  ax.plot     running mean of r*, per model
    zero_line     ax.axhline  r* = 0
    crit_lines    ax.axhline  r* = +/- z_crit (labelled on the first panel only)
    grid          ax.grid     keywords for the grid (None hides it)
    model_note    ax.text     the in/out note at the top left, just above the
                              frame; "s" is its text (a template, see
                              :data:`MODEL_NOTE_STYLE`)
    legend        ax.legend   keywords for the legend (None hides it)

    and two non-element ``style`` entries for the note:

    style["model_note_status"]      {"in" | "out" | "mixed": {kwarg: value} or None}
                                    -- every note of that status
    style["model_note_covariates"]  {covariate: {kwarg: value} or None}
                                    -- that panel's note only; applied last

    Returns
    -------
    fig : Figure
    residuals : DataFrame or dict
        :func:`whitened_residuals` of each model (computed once per model, not
        once per panel): a DataFrame for a single model, ``{model_id: DataFrame}``
        for a list.

    Raises
    ------
    KeyError
        A covariate is neither a column of the rows nor an index level. All are
        checked before anything is drawn.
    ValueError
        A covariate is not numeric, ``covariates`` is empty, or ``axes`` has too
        few axes.

    Warns
    -----
    UserWarning
        For a group-scope outcome (A2, one row per design group), for each
        covariate not in the model that is not design-level (see
        :data:`DESIGN_LEVEL_COVARIATES`): it carries the value of each group's
        first site.

    Example
    -------
    >>> covs = ["d_ln_ds595_mean_unif", "d_ln_ds595_sigma_unif",
    ...         "d_ln_saratio_mean_unif", "d_ln_saratio_sigma_unif",
    ...         "ln_T1", "ln_overstrength", "ln_ductility"]
    >>> for q in ("theta", "beta"):
    ...     fig, res = fmd.plot_whitened_residuals_vs_covariates(
    ...         joint["A8f_bc_hb"], q, covs,
    ...         style={"model_note_covariates": {"ln_T1": {"color": "0.4"}}})
    """
    views, single = _as_views(models, quantity)
    style = dict(style or {})
    st = _Styler(style, model_styles, views, colour_by)
    q = views[0].quantity
    z_crit = stats.norm.ppf(1 - significance_level / 2)
    multi = st.K > 1
    labels = dict(covariate_labels or {})

    # --- validate everything before any figure exists ----------------------
    covs = [covariates] if isinstance(covariates, str) else list(covariates)
    if not covs:
        raise ValueError("no covariates given")
    _check_covariates(views, covs)
    if axes is not None:
        # dtype=object so a 2-D array of Axes (or a single Axes) flattens to Axes.
        flat = list(np.asarray(axes, dtype=object).ravel())
        if len(flat) < len(covs):
            raise ValueError(f"{len(covs)} covariates need {len(covs)} axes, got {len(flat)}")
        fig = flat[0].figure
        nc = ncols
    _warn_group_scope(views, covs)

    # --- the expensive step, once per model: Sigma's eigendecomposition ------
    results = {vw.model_id: whitened_residuals(vw, method=method) for vw in views}

    # --- figure ---------------------------------------------------------------
    own_figure = axes is None
    if own_figure:
        nc = min(ncols, len(covs))
        nr = int(np.ceil(len(covs) / nc))
        fig, grid = plt.subplots(nr, nc, figsize=(figsize_per_panel[0] * nc,
                                                  figsize_per_panel[1] * nr),
                                 sharey=sharey, squeeze=False)
        flat = list(grid.ravel())

    # --- data: models outermost, as in plot_whitened_residuals ---------------
    for k, vw in enumerate(views):
        r = results[vw.model_id]["r_white"].to_numpy()
        w = running_window or max(5, r.size // 10)
        pre = "{model}: " if multi else ""
        for ax, cov in zip(flat, covs):
            x = vw.covariate_values(cov).astype(float)      # raw stored value (a log)
            _draw_r_vs_x(ax, st, k, vw, x, r, w, pre)

    # --- per-panel furniture, labels and the in/out note ---------------------
    for j, (ax, cov) in enumerate(zip(flat, covs)):
        # Label the crit lines on the first panel only: that is where the one
        # legend goes, and it then gets a single entry for the pair.
        _residual_axes_furniture(ax, st, z_crit, label_crit=(j == 0))
        ax.set_xlabel(labels.get(cov, cov))
        if not sharey or _is_first_column(ax, j, nc):
            ax.set_ylabel(r"whitened residual $r^*$")

        in_ids = [v.model_id for v in views if cov in v.x_names]
        out_ids = [v.model_id for v in views if cov not in v.x_names]
        status = "in" if not out_ids else ("out" if not in_ids else "mixed")
        letter = _panel_letter(j) if panel_letters else ""
        kw = _model_note_kw(style, status, cov) if model_note else None
        if kw is not None:
            text = str(kw.pop("s", "")).format_map(_Fmt(
                {"in_models": ", ".join(in_ids), "out_models": ", ".join(out_ids),
                 "covariate": cov}))
            text = f"{letter}  {text}" if letter else text
            # Anchored at the frame's top-left corner and sitting just above it,
            # where set_title(loc="left") would draw -- hence no left title here.
            # ax.text artists are part of the tight bbox, so tight_layout makes
            # room for the note.
            ax.text(0.0, 1.01, text, transform=ax.transAxes, **kw)
        elif letter:
            ax.set_title(letter, loc="left", fontsize=10)

    if legend:
        kw = st.kw("legend", "dict", {"fontsize": 7, "loc": "lower right"})
        if kw is not None:
            flat[0].legend(**kw)

    if own_figure:
        for ax in flat[len(covs):]:
            ax.set_axis_off()
        ids = [vw.model_id for vw in views]
        who = ", ".join(ids) if len(ids) <= 3 else f"{len(ids)} models"
        fig.suptitle(f"{who}: whitened residuals vs covariates ({q})", fontsize=16)
        fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, facecolor="w")

    return fig, (results[views[0].model_id] if single else results)


# =============================================================================
# All three diagnostics for one model
# =============================================================================

#: Axis labels for the covariates, used by :func:`plot_model_diagnostics`.
#: A covariate not listed here is labelled with its column name.
DEFAULT_COVARIATE_LABELS: dict[str, str] = {
    "d_ln_saratio_mean_unif": "Ratio ofSARatio($T_1$) Means: SS/FX",
    "d_ln_ds595_mean_unif": "Ratio of Duration Means: SS/FX",
    "d_ln_ds595_sigma_unif": "Ratio of Duration Std. Deviations: SS/FX",
    "d_ln_saratio_sigma_unif": "Ratio ofSARatio($T_1$) Std. Deviations: SS/FX",
    "ln_T1": "Fundamental Period, $T_1$ [s]",
    "ln_overstrength": r"Overstrength,  $\Omega = V_{max} / V_{design}$",
    "ln_ductility": r"Ductility, $\mu$",
}

#: Covariates plotted for a model that has NONE of its own, by contrast scope.
#: The conditional-effect panel then shows a flat line at the model's mean with
#: the observed points against that covariate -- i.e. whether there is a trend
#: the model leaves out. Row scope (120 rows) uses the two site-level duration
#: ratios; group scope (51 designs) can only use design-level quantities,
#: because a site covariate has no single value for a design group.
DEFAULT_FALLBACK_COVARIATES: dict[str, tuple[str, ...]] = {
    "row": ("d_ln_ds595_mean_unif", "d_ln_ds595_sigma_unif"),
    "group": ("ln_T1",),
}


def _is_ratio_covariate(name: str) -> bool:
    """True for a covariate that is a log RATIO (SS/FX), where x = 1 is a reference."""
    return name.startswith("d_ln_")


def plot_model_diagnostics(model, quantities: str | Sequence[str] | None = None, *,
                           covariates: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
                           fallback_covariates: Mapping[str, Sequence[str]] | None = None,
                           sort_covariate: Mapping[str, str] | str | None = None,
                           covariate_labels: Mapping[str, str] | None = None,
                           significance_level: float = 0.05, ncols: int = 3,
                           whitened_kw: Mapping | None = None,
                           obs_fit_kw: Mapping | None = None,
                           conditional_kw: Mapping | None = None,
                           save_dir=None, dpi: int = 150) -> dict:
    """Whitened residuals, observed vs fitted and conditional effects for ONE model.

    A convenience wrapper that lays out the three diagnostic plots of this
    module for a single fit, so that one line after each model's summary gives
    the full set. Everything it draws comes from :func:`plot_whitened_residuals`,
    :func:`plot_observed_vs_fitted` and :func:`plot_conditional_effect`; their
    own options (including ``style`` and ``model_styles``) are reached through
    ``whitened_kw``, ``obs_fit_kw`` and ``conditional_kw``.

    Figures produced (keys of the returned dict):

        "whitened"             one row of three panels per quantity
        "observed_vs_fitted"   one panel per quantity, square, with the 1:1 line
        "conditional_<q>"      one panel per covariate of quantity q

    Which covariates the conditional-effect panels show, per quantity:
      1. ``covariates`` if given;
      2. otherwise the covariates IN THE MODEL for that quantity;
      3. for an intercept-only model, ``fallback_covariates[scope]`` that exist
         in the contrast rows (see :data:`DEFAULT_FALLBACK_COVARIATES`). Those
         panels are titled "(not in model)": the line is flat at the model mean,
         and the points show whether the data carry a trend the model omits.
    If none of these gives a covariate, the conditional figure for that quantity
    is not drawn.

    Parameters
    ----------
    model : ModelFit or JointModelFit
    quantities : "theta", "beta", a list of them, or None
        None takes the fit's own quantity (univariate) or both (joint).
    covariates : list of str, or {quantity: list of str}, optional
        Override the conditional-effect covariates. A plain list applies to
        every quantity.
    fallback_covariates : {scope: list of str}, optional
        Override :data:`DEFAULT_FALLBACK_COVARIATES`.
    sort_covariate : str or {quantity: str}, optional
        x-axis of the third whitened-residual panel. Default: the first
        conditional-effect covariate of that quantity, else "n_storeys".
    covariate_labels : {covariate: label}, optional
        Merged over :data:`DEFAULT_COVARIATE_LABELS`.
    significance_level : float
    ncols : int
        Panels per row in the conditional-effect figures.
    whitened_kw, obs_fit_kw, conditional_kw : dict, optional
        Extra keyword arguments for the three plot functions, e.g.
        ``whitened_kw={"stats_items": ["Q", "r_lims"]}`` or
        ``conditional_kw={"style": {"pi_band": None}}``.
    save_dir : path, optional
        If given, every figure is saved there as ``<model_id>_<key>.png``.
    dpi : int
        Resolution of the saved PNGs.

    Returns
    -------
    dict
        ``{key: Figure}`` as listed above, plus ``"residuals"``:
        ``{quantity: DataFrame}`` from :func:`whitened_residuals`.
    """
    from pathlib import Path
    from matplotlib.ticker import FormatStrFormatter

    # --- which quantities --------------------------------------------------
    is_joint = hasattr(model, "T_total") and hasattr(model.contrast, "theta")
    if quantities is None:
        qs = list(fm.OUTCOMES) if is_joint else [model.quantity]
    elif isinstance(quantities, str):
        qs = [quantities]
    else:
        qs = list(quantities)
    views = {q: fit_view(model, q) for q in qs}

    labels = {**DEFAULT_COVARIATE_LABELS, **(covariate_labels or {})}
    fallback = {**DEFAULT_FALLBACK_COVARIATES, **(fallback_covariates or {})}

    # --- conditional-effect covariates per quantity -------------------------
    cond_covs: dict[str, list[tuple[str, bool]]] = {}     # q -> [(covariate, in_model)]
    for q, vw in views.items():
        own = [nm for nm in vw.x_names if nm != "intercept"]
        if covariates is not None:
            chosen = list(covariates.get(q, []) if isinstance(covariates, Mapping) else covariates)
        elif own:
            chosen = own
        else:
            chosen = [c for c in fallback.get(vw.contrast.scope, ())
                      if c in vw.contrast.rows.columns]
        cond_covs[q] = [(c, c in own) for c in chosen]

    def _sort_cov(q: str) -> str:
        if isinstance(sort_covariate, str):
            return sort_covariate
        if isinstance(sort_covariate, Mapping) and q in sort_covariate:
            return sort_covariate[q]
        return cond_covs[q][0][0] if cond_covs[q] else "n_storeys"

    figs: dict = {}
    mid = model.model_id

    # --- 1. whitened residuals: one row of three panels per quantity --------
    fig, axes = plt.subplots(len(qs), 3, figsize=(15, 4.5 * len(qs)), squeeze=False)
    residuals = {}
    for ax_row, q in zip(axes, qs):
        residuals[q] = plot_whitened_residuals(ax_row, views[q], None, _sort_cov(q),
                                               significance_level=significance_level,
                                               **(whitened_kw or {}))
        # The third panel's x-axis is the covariate as stored (a log), so it keeps
        # its column name as the label rather than the ratio-scale label.
    axes[0, 1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{mid}: whitened residuals", fontsize=16)
    fig.tight_layout()
    figs["whitened"] = fig

    # --- 2. observed vs fitted: one square panel per quantity ---------------
    fig, axes = plt.subplots(1, len(qs), figsize=(5.5 * len(qs), 5.5), squeeze=False)
    for ax, q in zip(axes[0], qs):
        vw = views[q]
        plot_observed_vs_fitted(ax, vw, None, **(obs_fit_kw or {}))
        # Equal, padded limits on both axes so the 1:1 line is the diagonal.
        vals = np.exp(np.concatenate([vw.y_obs, vw.fitted]))
        lo, hi = vals.min(), vals.max()
        pad = 0.05 * (hi - lo) if hi > lo else 0.05
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axisbelow(True)
        ax.grid(ls="-.", color="0.8")
        ax.set_xlabel(ratio_label(q, "Observed"), fontsize=12)
        ax.set_ylabel(ratio_label(q, "Fitted"), fontsize=12)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{mid}: fitted vs observed", fontsize=16)
    fig.tight_layout()
    figs["observed_vs_fitted"] = fig

    # --- 3. conditional effects: one figure per quantity --------------------
    for q in qs:
        covs = cond_covs[q]
        if not covs:
            continue
        nc = min(ncols, len(covs))
        nr = int(np.ceil(len(covs) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(5.3 * nc, 4.5 * nr), sharey=True,
                                 squeeze=False)
        flat = axes.flatten()
        for ii, (ax, (cov, in_model)) in enumerate(zip(flat, covs)):
            ax.set_axisbelow(True)
            ax.grid(ls="-.", color="0.8")
            kw = {"vline_at_one": _is_ratio_covariate(cov), **(conditional_kw or {})}
            plot_conditional_effect(ax, views[q], None, cov, significance_level, **kw)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.xaxis.set_minor_locator(plt.NullLocator())
            ax.set_xlabel(labels.get(cov, cov), fontsize=11)
            if ii % nc == 0:
                ax.set_ylabel(ratio_label(q), fontsize=12)
            ax.set_title(f"({chr(65 + ii)})" + ("" if in_model else "  (not in model)"),
                         loc="left", fontsize=10)
        for ax in flat[len(covs):]:
            ax.set_axis_off()
        fig.suptitle(rf"{mid}: conditional effects, $\{q}$", fontsize=16)
        fig.tight_layout()
        figs[f"conditional_{q}"] = fig

    if save_dir is not None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for key, f in figs.items():
            f.savefig(out / f"{mid}_{key}.png", dpi=dpi, facecolor="w")

    figs["residuals"] = residuals
    return figs
