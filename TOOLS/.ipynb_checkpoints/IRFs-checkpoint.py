import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from datetime import datetime

def shock_generator(ss, var, rho, T, truncated_T, p=1/100, bp=False, display=True, title='', tol=1e-14):
    """Generates a shock.
    
    Inputs
    -----------------
    ss               : dict, steady-state of the model
    var              : str, variable on which to make the shock
    rho              : float, persistence of the shock
    T                : int, time horizon for shock
    truncated_T      : int, time horizon for plot
    p                : float, size of a shock (1/100 for a 1% positive shock)
    bp               : bool, True if expressed in basis point (otherwise percentage point)
    display          : bool, True to plot the shock
    title            : str, name of the shock
    tol              : float, below this value, the steady-state value is considered 0
    
    Outputs
    -----------------
    dZ               : array (T), shock
    """
    # Generate the shock
    if ss[var] > tol:
        dZ = p*ss[var]*rho**(np.arange(T)[:, np.newaxis])
    else:
        dZ = p*rho**(np.arange(T)[:, np.newaxis])
    
    if bp:
        title2 = 'Basis point'
        coeff = 10000
    else:
        title2 = 'Percentage point'
        coeff = 100
    
    if display:
        ax = plt.figure().gca()
        if ss[var] > 0:
            plt.plot(coeff*dZ[:truncated_T]/ss[var])
            plt.ylabel(title2 + ' deviations from steady-state')
        else:
            plt.plot(coeff*dZ[:truncated_T])
            plt.ylabel(title2 + ' differences from steady-state')
        plt.plot([0]*truncated_T, 'k--')
        plt.title(title + ' shock')
        plt.rc('text', usetex=False)
        plt.rc('font', family='serif')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()
        
    return dZ
    
    
def plot_IRFs(G, ss, shock, var_shock, var, var_names, T, truncated_T, bp=False, level=True, title='', save=False, save_fig=False, DPI=500, tol=10e-14):
    """Plots the IRFs.
    Inputs
    -----------------
    G                : dict, Jacobian
    ss               : dict, steady state
    shock            : array, shock
    var_shock        : str, key of the shock
    var              : list, IRFs to compute
    var_names        : list, Names to display
    T                : int, time horizon
    truncated_T      : int, time horizon for plot
    bp               : bool, True if expressed in basis point (otherwise percentage point)
    title            : str, characteristics of the shock to display
    save             : bool, True if want to save IRFs to csv file
    save_fig         : bool, True if want to save the plot
    DPI              : int, quality of the figure
    tol              : float, consider null response below this threshold
    
    Outputs
    -----------------
    df_IRFs          : DataFrame, IRFs computed
    """
    nb_var = len(var) # Number of variables
    
    if bp:
        title2 = 'Basis point'
        coeff = 10000
    else:
        title2 = 'Percentage point'
        coeff = 100
    
    # Compute the IRFs for all variables
    IRFs = {}
    if level:
        for i in range(nb_var):
            if np.max(np.abs(G[var[i]][var_shock] @ shock)) < tol:
                dvar = np.zeros(T)
                IRFs[var[i]] = dvar
            else:
                assert ss[var[i]] != 0, "Cannot divide by null steady-state value"
                dvar = G[var[i]][var_shock] @ shock / ss[var[i]]
                dvar = coeff*dvar
                IRFs[var[i]] = dvar.reshape(T)
    else:
        for i in range(nb_var):
            if np.max(np.abs(G[var[i]][var_shock] @ shock)) < tol:
                dvar = np.zeros(T)
            else:
                dvar = G[var[i]][var_shock] @ shock
            dvar = coeff*dvar
            IRFs[var[i]] = dvar.reshape(T)
    
    # Plot the IRFs
    if nb_var%3 != 0:
        nb_lines = (nb_var//3) + 1
    else:
        nb_lines = nb_var//3

    fig, axs = plt.subplots(nb_lines, 3, figsize=(12,nb_lines*3))
    if nb_lines == 1:
        for i in range(3):
            if i < nb_var:
                axs[i].plot(IRFs[var[i]][:truncated_T], 'tab:red', label=var_names[i])
                axs[i].plot([0]*truncated_T, 'k--')
                axs[i].legend(frameon=False,handlelength=0, handletextpad=0,)
            else: # Delete useless subplots
                fig.delaxes(axs[i])
    else:
        for i in range(nb_lines*3):
            q = i//3
            r = i%3
            if i < nb_var:
                axs[q,r].plot(IRFs[var[i]][:truncated_T], 'tab:red', label=var_names[i])
                axs[q,r].plot([0]*truncated_T, 'k--')
                axs[q,r].legend(frameon=False,handlelength=0, handletextpad=0,)
            else: # Delete useless subplots
                fig.delaxes(axs[q,r])

    # Add titles
    if nb_lines == 1:
        fontsize = 'large'
        fig.text(0.5, 0.0, 'Quarter', ha='center', fontsize=fontsize)
    else:
        fontsize = 'x-large'
        fig.text(0.5, 0.05, 'Quarter', ha='center', fontsize=fontsize)
    if level:
        fig.text(0.05, 0.5, title2 + ' deviations from steady-state', va='center', rotation='vertical', fontsize=fontsize)
    else:
        fig.text(0.05, 0.5, title2 + ' differences to steady-state', va='center', rotation='vertical', fontsize=fontsize)
    fig.text(0.5, 0.9, 'Impulse response functions to a ' + title + ' shock', ha='center', fontsize='xx-large')
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    
    if save_fig:
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        if level:
            filename = 'IRFs_level_' + title + "_" + date + ".png" 
        else:
            filename = 'IRFs_' + title + "_" + date + ".png" 
        plt.savefig(filename, format='png', dpi=DPI)
    plt.show()
    
    IRFs['shock'] = coeff*shock.reshape(T)
    df_IRFs = pd.DataFrame(IRFs)
    if save:
        df_IRFs.to_csv('IRF_' + title + '.csv', sep=';') 
    
    return df_IRFs
