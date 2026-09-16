import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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