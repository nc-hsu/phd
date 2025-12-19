"""Script to calculate risk contributions for Tailfingen considering
different target mean annual frequencies of collapse (MAFC).

The hazard curves are assumed to be linear in log-log space.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from matplotlib.ticker import ScalarFormatter

# Risk parameters
target_mafcs = [2e-4, 5e-05, 1e-5]   # target mean annual frequency of collapse

# hazard parameters first order -> Tailfingen for SA = 0.5s on rock
k0 = 6.614486E-05
k1 = 1.76677717
max_sa = 15  # maximum spectral acceleration value
step = 0.1

# fragility parameters
beta = 0.4  # fragility curve dispersion

# define the hazard curve
sa = np.linspace(0, max_sa, int(max_sa/step))  # spectral acceleration values
sa[0] = 1e-6  # avoid log(0) issues
H_im = k0 * sa**(-k1)  # hazard curve in log-log space

# define derivative of hazard curve
dH_dx = -k0 * k1 * sa**(-k1 - 1)

def theta_c_from_mafc_first_order(target_mafc, k0, k1, beta):
    """Calculate the required theta_c for a given target MAFC."""
    return np.exp((0.5 * k1 * beta ** 2) - (np.log(target_mafc / k0) / k1))

def fragility_curve(sa, theta_c, beta):
    """Calculate the fragility curve."""
    return lognorm.cdf(sa, beta, scale=theta_c)

def analytical_mafc_first_order(theta_c, k0, k1, beta):
    """Calculate the analytical mean annual frequency of collapse."""
    return k0 * theta_c ** (-k1) * np.exp(0.5 * (beta * k1) ** 2)

def numerical_mafc(sa, dH_dx, fragility):
    """Calculate the numerical mean annual frequency of collapse."""
    y = fragility * np.abs(dH_dx)
    return np.trapz(y, sa)

def risk_contribution(fragility, dH_dx, step, mafc):
    """Calculate the risk contribution."""
    return fragility * np.abs(dH_dx) * step / mafc

# Plot the risk contributions for each target MAFC
plt.figure()
ax = plt.gca()

for mafc_t in target_mafcs:
    theta_c = theta_c_from_mafc_first_order(mafc_t, k0, k1, beta)
    fragility = fragility_curve(sa, theta_c, beta)
    
    mafc_a = analytical_mafc_first_order(theta_c, k0, k1, beta)
    mafc_n = numerical_mafc(sa, dH_dx, fragility)
    
    risk_contribution_values = risk_contribution(fragility, dH_dx, step, mafc_n)

    print(40*"=")
    print(f"Target MAFC: {mafc_t:.3e}")
    print(f"Required theta_c for target MAFC: {theta_c:.3f}")
    print(f"MAFC (analytical): {mafc_a:.3e}")
    print(f"MAFC (numerical): {mafc_n:.3e}")
    print(40*"=")

    ax.bar(sa, risk_contribution_values, width=step*1.01, \
           label=f'$\lambda_{{c,t}}$ = {mafc_t:.1e}', align='edge', alpha=0.7)

ax.legend(title='Target MAFC', loc='upper right')
ax.set_xlabel('Spectral Acceleration (g)')
ax.set_ylabel('Risk Contribution')
ax.set_xlim(0, 8)

# plot hazard and fragility curves
fig, ax1 = plt.subplots() 
lns = []
ln, = ax1.plot(sa, H_im, label='Hazard', color='k')
lns.append(ln)
ax2 = ax1.twinx()

for mafc_t in target_mafcs:
    theta_c = theta_c_from_mafc_first_order(mafc_t, k0, k1, beta)
    fragility = fragility_curve(sa, theta_c, beta)
    ln, = ax2.plot(sa, fragility, label=f'$\lambda_{{c,t}}$ = {mafc_t:.1e}')
    lns.append(ln)

labs = [l.get_label() for l in lns]

ax1.legend(lns, labs)
ax1.set_xlabel('Spectral Acceleration (g)')
ax1.set_ylabel('Fragility Curve')
ax1.set_xlim(0, 8)

ax1.set_ylim(0, 1e-3)
ax2.set_ylim(0, 1)


# plt hazard curves and fragility curves together with risk contributions

for mafc_t in target_mafcs:
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax3 = ax1.twinx() 
    
    theta_c = theta_c_from_mafc_first_order(mafc_t, k0, k1, beta)
    fragility = fragility_curve(sa, theta_c, beta)
    mafc_n = numerical_mafc(sa, dH_dx, fragility)
    risk_contribution_values = risk_contribution(fragility, dH_dx, step, mafc_n)

    ln1, = ax1.plot(sa, H_im, label='Hazard', color='k')
    ln2, = ax2.plot(sa, fragility, label=f'Fragility Curve') 
    ln3 = ax3.bar(sa, risk_contribution_values, width=step*1.01, 
                   label=f'Risk Contribution', alpha=0.7, align='edge')
    
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # always use scientific notation
    ax1.yaxis.set_major_formatter(formatter)

    lines = [ln1, ln2, ln3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    ax1.set_xlabel('Spectral Acceleration (g)')
    ax1.set_ylabel('Hazard - MAFE [1/yr]')
    ax2.set_ylabel('Fragility Curve - P[Collapse | SA]')
    ax3.set_ylabel('Risk Contribution [-]')
    ax3.spines["right"].set_position(("outward", 60))
    ax3.spines["right"].set_visible(True)
    ax1.set_xlim(0, 8)
    ax1.set_ylim(0, 1e-3)
    ax2.set_ylim(0, 1)
    ax3.set_ylim(0,0.2)
    fig.subplots_adjust(right=0.75)

    plt.title(f"Risk Contribution for $\lambda_{{c,t}}$ = {mafc_t:.1e}")



plt.show()


