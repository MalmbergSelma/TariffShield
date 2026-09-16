import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

def get_cumulated_array(data, **kwargs):
    cum = data.clip(**kwargs)
    cum = np.cumsum(cum, axis=0)
    d = np.zeros(np.shape(data))
    d[1:] = cum[:-1]
    return d 
    

def plot_shock_decomposition_1axis(Ds, data, shocks, sh_dict, variables, var_dict, xaxis, savefig=False, name='Figure.eps'):
    """Plot the shock decomposition"""
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    T, nb_var, nb_shock = Ds.shape
    nb_lines = nb_var//3 + min(nb_var%3, 1)
    heights  = [3] * nb_lines + [1] 
    
    # Plot
    fig = plt.figure(constrained_layout=True, figsize=(15, nb_lines*4))
    gs  = fig.add_gridspec(nb_lines+1, 3, height_ratios=heights)
    
    for i in range(nb_var):
        idx = i//3, i%3
        # Compute the cumulated contributions
        cumulated_data_pos = get_cumulated_array(Ds[:, i, :].transpose(), min=0)
        cumulated_data_neg = get_cumulated_array(Ds[:, i, :].transpose(), max=0)
        row_mask = (Ds[:, i, :].transpose()<0)
        cumulated_data_pos[row_mask] = cumulated_data_neg[row_mask]
        data_stack = cumulated_data_pos
        # Plot the contributions
        ax  = fig.add_subplot(gs[idx])
        for z in np.arange(0, nb_shock):
            if z == 0:
                ax.bar(xaxis, Ds[:, i, :].transpose()[z], bottom=data_stack[z], color=sh_dict[shocks[z]]['color'], 
                       width=xaxis[1]-xaxis[0], label=var_dict[variables[i]]['label'])
            else:
                ax.bar(xaxis, Ds[:, i, :].transpose()[z], bottom=data_stack[z], color=sh_dict[shocks[z]]['color'], 
                       width=xaxis[1]-xaxis[0])
        # Plot the aggregate series
        ax.plot(xaxis, data[:, i], 'black', linewidth=1.5)
        ax.axhline(0, color='black', linestyle='--', alpha=0.4)
        ax.legend(frameon=False, handlelength=0, handletextpad=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Legend
    cust_leg = []
    for i in range(nb_shock):
        cust_leg.append(mpatches.Patch(color=sh_dict[shocks[i]]['color'], label=sh_dict[shocks[i]]['label']))
    cust_leg.append(Line2D([0], [0], label='Observed time series', color='black'))
    ax = fig.add_subplot(gs[-1,:])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(handles=cust_leg, loc='center', frameon=False, ncol=4)
    
    plt.tight_layout()
    if savefig:
        plt.savefig(name, dpi=600)
    plt.show()


def generate_M(inputs, new_outputs, sigmas, rhos, T, G, verbose=False):
    new_impulses = {}
    for i in inputs:
        own_shock = sigmas[i] * rhos[i]**np.arange(T)
        interm = {}
        for j in new_outputs:
            if verbose:
                print(i, j)
            if j in inputs:
                if i == j:
                    interm[j] = np.eye(T) @ own_shock
                else:
                    interm[j] = np.zeros(T)
            else:
                interm[j] = G[j][i] @ own_shock
        interm[i] = own_shock
        new_impulses[i] = interm
    
    new_M = np.empty((T, len(new_outputs), len(inputs)))
    for no, o in enumerate(new_outputs):
        for ns, s in enumerate(inputs):
            new_M[:, no, ns] = new_impulses[s][o]
    
    return new_M


def reconstruct(As, eps_hat):
    """Calculates most likely shock paths if As is true set of IRFs

    Parameters
    ----------
    As : array (Tm*O*E) giving the O*E matrix mapping shocks to observables at each of Tm lags in the MA(infty),
            e.g. As[6, 3, 5] gives the impact of shock 5, 6 periods ago, on observable 3 today
    eps_hat : array (To*E) giving most likely path of all shocks

    Returns
    ----------
    Ds : array (To*O*E) giving the level of each observed data series that is accounted for by each shock
    """
    Tm, O, E = As.shape
    To = eps_hat.shape[0]

    A_full = construct_stacked_A(As, To)

    # Step 3: Decompose data
    for e in range(E):
        A_full = A_full.reshape((To,O,To,E))
        Ds = np.sum(A_full * eps_hat, axis=2)

    return Ds


def construct_stacked_A(As, To, sigma_e=None, sigma_o=None, reshape=True, long=False):
    Tm, O, E = As.shape
    To_out = To

    # allocate memory for A_full
    A_full = np.zeros((To_out, O, To, E))

    for o in range(O):
        for itshock in range(To):
            # if To > To_out, allow the first To - To_out shocks to happen before the To_out time periods
            if To <= To_out:
                iA_full = itshock
                iAs = 0

                shock_length = min(Tm, To_out - iA_full)
            else:
                # this would be the correct start time of the shock
                iA_full = itshock - (To - To_out)

                # since it can be negative, only start IRFs at later date
                iAs = - min(iA_full, 0)

                # correct iA_full by that date
                iA_full += - min(iA_full, 0)

                shock_length = min(Tm, To_out - iA_full)

            for e in range(E):
                A_full[iA_full:iA_full + shock_length, o, itshock, e] = As[iAs:iAs + shock_length, o, e]
                if sigma_e is not None:
                    A_full[iA_full:iA_full + shock_length, o, itshock, e] *= sigma_e[e]
                if sigma_o is not None:
                    A_full[iA_full:iA_full + shock_length, o, itshock, e] /= sigma_o[o]
    if reshape:
        A_full = A_full.reshape((To_out * O, To * E))
    return A_full
    

def generate_draws(eps_hat_est_df, N, T, labels, sigmas=None, mus=None, CI=95, display=False):
    """Generate N shock paths starting from the eps_hat_est_df estimates.
    For the first T periods, no alternative path is possible."""
    # Get the standard deviations of the estimated shock series
    if sigmas is None:
        sigmas = eps_hat_est_df.loc[eps_hat_est_df.index[9:]].std()
    if mus is None:
        mus    = eps_hat_est_df.loc[eps_hat_est_df.index[9:]].mean()
    # Initialisation
    nb_periods, nb_shock = eps_hat_est_df.shape
    nb_shock -= 1 # No uncertainty on the subv shock
    nb_rows = nb_shock//3
    if nb_shock%3 != 0:
        nb_rows += 1
    results = {}
    
    if display:
        fig, axs = plt.subplots(nb_rows, 3, figsize=(5*3, nb_rows*5))
    # Loop on all shocks
    for (c, col) in enumerate(eps_hat_est_df.columns[:-1]):  # No uncertainty on subv shock
        val = np.random.normal(loc=mus[c], scale=sigmas[c], size=(N, nb_periods))
        val2 = np.zeros((int(CI*N/100), nb_periods))
        for t in range(nb_periods):
            x = (100-CI)/2
            perc1, perc2 = np.percentile(val[:, t], x), np.percentile(val[:, t], 100-x)
            interm = val[:, t]
            val2[:, t] = interm[(interm >= perc1) & (interm <= perc2)] # Drop extreme values
        
        for i in range(int(CI*N/100)):
            val2[i, :T] = eps_hat_est_df[col][eps_hat_est_df.index[:T]]
            if display:
                axs[c//3, c%3].plot(eps_hat_est_df.index, val2[i], color='lightgrey')
        results[col] = val2
        if display:
            axs[c//3, c%3].plot(eps_hat_est_df.index, eps_hat_est_df[col], color='black', label='Benchmark')
            axs[c//3, c%3].plot(eps_hat_est_df.index, val2.mean(axis=0), color='red', label='Average')
            axs[c//3, c%3].legend(frameon=False, title=labels[c])
    if display:
        plt.show()
    return results

    
def rebuild_level(stat_serie, name, T, g=0, a=0):
    """g is the trend's growth rate
    a is the historical average"""
    if name in ['Output', 'Consumption', 'Energy firm', 'Energy household', 'Transfers', 'Government spending']:
        level = (1 + stat_serie) * (1+g)**(np.arange(T)) 
        level = (100 * level) / level[0]
    elif name in ['Real rate', 'Nominal rate', 'Real wages', 'Labor']:
        level = 100 * stat_serie
    elif name == 'Debt-to-GDP ratio':
        level = 100 * (stat_serie + 4) / 4    
    elif name == 'Inflation':
        level = 100 * (stat_serie + a)   
    elif name == 'Prices':
        level = [100]
        for i in range(T-1):
            level.append(level[i] * (1 + stat_serie[i+1]))
    return level