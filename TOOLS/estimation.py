import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, invgamma, gamma, norm
import sequence_jacobian as ssj

def marginal_prior_distribution(distrib, params):
    """Builds the prior distribution function.

    Parameters
    ----------
    distrib    : str, name of the distribution considered
    params     : array, parameters of the distribution (mean and standard deviation)

    Returns
    ----------
    x          : array, domain of the prior distribution 
    prior      : array, prior distribution
    """
    if distrib == 'beta':
        mean_beta,std_beta = params
        if mean_beta <= 0:
            raise ValueError('Mean must be positive.')
        if std_beta <=0:
            raise ValueError('Standard deviation must be positive.')
        quad = ((mean_beta**2) - mean_beta + (std_beta**2))
        a = - (mean_beta * quad)/(std_beta**2)
        b = ((mean_beta - 1)*quad)/(std_beta**2)
        x = np.linspace(0, 1, 1000)
        return x, beta.pdf(x, a, b)
    
    elif distrib == 'invgamma':
        mean_invgam,std_invgam = params
        if mean_invgam <= 0:
            raise ValueError('Mean must be positive.')
        if std_invgam <= 0:
            raise ValueError('Standard deviation must be positive.')
        a = 2 + (mean_invgam / std_invgam)**2
        scale = mean_invgam + ((mean_invgam**3)/(std_invgam**2))
        x = np.linspace(0, 10*mean_invgam, 1000)
        return x, invgamma.pdf(x=x, a=a, scale=scale) 
    
    elif distrib == 'gamma':
        mean_gam,std_gam = params
        if mean_gam <= 0:
            raise ValueError('Mean must be positive.')
        if std_gam <=0:
            raise ValueError('Standard deviation must be positive.')
        scale = (std_gam**2)/mean_gam
        a = mean_gam/scale
        x = np.linspace(0, 3*mean_gam, 1000)
        return x, gamma.pdf(x=x, a=a, scale=scale)
    
    elif distrib == 'normal':
        mean_nor,std_nor = params
        loc = mean_nor
        scale = std_nor
        x = np.linspace(mean_nor-3*std_nor, mean_nor+3*std_nor, 1000)
        return x, norm.pdf(x=x, loc=loc, scale=scale)
        
    else:
        return None
        


def transition(x, prior, c=None, covr=None):
    """Draw a sample from a distribution and returns it.
    
    Parameters
    ----------
    x         : array (N), current draw
    prior     : function, computes the prior probability
    c         : float, scaling factor
    covr      : array (N,N), variance-covariance matrix
    
    Returns
    ----------
    x_new      : array (N), new draw"""
    n = len(x)
    mean = [0] * n
    if c is None:
        c = 1
    if covr is None:
        covr = np.eye(n)
    x_new = (x + np.random.multivariate_normal(mean, c*covr, 1)).reshape(n,)
    return x_new


def acceptance(x, x_new):
    """Boolean function that states wether a new sample should be accepted or not"""
    if x_new > x:
        return True
    else:
        accept = np.random.uniform(0,1)
        return (accept < (np.exp(x_new-x)))


def metropolis_hastings(log_like, prior, param_init, iterations, data, c=None, covr=None, saveReject=False, display=False):
    """Metropolis-Hastings algorithm
    
    Parameters
    ----------
    log_like    : function, computes log-likelihood given parameters and data
    prior       : function, prior density for given parameters
    param_init  : array, starting sample
    iterations  : int, number of draws
    data        : array, observations
    saveReject  : bool, True if wish to save the rejected draws
    display     : bool, True if want to display indication of the number of iterations

    Returns
    ----------
               : array, """
    # Initialisation
    x = param_init
    accepted = []
    if saveReject:
        rejected = []
    
    coeff = iterations//100
    for i in range(iterations):
        x_new = transition(x, prior, c, covr)
        x_lik = log_like(x, data)
        x_new_lik = log_like(x_new, data)
        
        # Display how iterations are going
        if i%coeff == 0 and display:
            if i == 0 or len(accepted) == 0:
                print('Starting point x = {}'.format(x), '; Log-likelihood {}'.format(x_lik))
            else:
                print('Iteration {}'.format(i), '; Log-likelihood {}'.format(last_accepted_log))
       
       # Accepted draw
        if (acceptance(x_lik + np.log(prior(x)), x_new_lik + np.log(prior(x_new)))):
            x = x_new
            last_accepted_log = log_like(x_new, data)
            accepted.append(x_new)
        # Rejected draw
        else:
            if saveReject:
                rejected.append(x_new)
    
    if saveReject:
        return np.array(accepted), np.array(rejected)
    else:
        return np.array(accepted)
    

def plot_posterior_sample(accepted_draws, colors, legends, nb_bins=100, save_fig=False, filename='BayesianEstimation.png', DPI=500):
    """Plot sample distribution of the accepted draws."""
    modes = []
    nb_params = accepted_draws.shape[1]
    nb_rows = (nb_params//3) + 1
    
    plt.figure(figsize=(10, 3*nb_rows))
    for i in range(nb_params):
        plt.subplot(nb_rows, 3, i+1)
        n, bins, patches = plt.hist(accepted_draws[:, i], bins=nb_bins, color=colors[1], density=True, stacked=True)
        mode_index = n.argmax()
        plt.axvline((bins[mode_index] + bins[mode_index+1])/2, c=colors[6], label='Mode')
        modes.append((bins[mode_index] + bins[mode_index+1])/2)
        plt.legend(frameon=False)
        plt.title(legends[i])
        plt.tight_layout()
    
    if save_fig:
        plt.savefig(filename, dpi=DPI)
    
    return modes


def likelihood_computation(var, shocks, persist, stdev, measurement, Tobs, G, Y, inout):
    """Compute the log-likelihood of the model.
    
    Parameters
    ----------
    var         : array (N), variables of the model to study
    shocks      : array (Z), shocks considered
    persist     : array (Z), persistence characterising the shocks
    stdev       : array (Z), standard deviation of each shock
    measurement : array (Z), measurement errors
    Tobs        : int, number of observations
    G           : dict, Jacobians of the model
    Y           : array (T,Nobs), observations
    inout       : array, variables that are inputs and outputs
    
    Returns
    ----------
    L           : float, log-likelihood"""
    NV = len(var)
    NS = len(shocks)

    # Generate the shocks
    dS = persist ** np.repeat(np.arange(Tobs).reshape(Tobs, 1), NS, axis=1)
    
    # Compute the system's responses and stack them in the matrix M
    dX = []
    for i in range(NS):
        dresponse = []
        for j in range(NV):
            if var[j] in inout:
                if shocks[i] in G[var[j]+'_out'].keys():
                    dY = G[var[j]+'_out'][shocks[i]].matrix(Tobs) @ dS[:, i] 
                else:
                    dY = np.zeros((Tobs, Tobs)) @ dS[:, i] 
            else:
                dY = G[var[j]][shocks[i]] @ dS[:, i] 
            dresponse.append(dY)
        dresponse = np.stack(dresponse, axis=1)
        dX.append(dresponse)
    M = np.stack(dX, axis=2)
    
    # Compute the covariances
    sigmas = np.array(stdev)
    Sigma = ssj.estimation.all_covariances(M, sigmas)
    
    # Return the log-likelihood
    return ssj.estimation.log_likelihood(Y, Sigma, measurement)
    
    
def gradient_f(x, f):
    """Computes the gradient of f at point x."""
    assert (x.shape[0] >= x.shape[1]), "the vector should be a column vector"
    N = x.shape[0]
    gradient = []
    for i in range(N):
        eps = abs(x[i]) * np.finfo(np.float32).eps 
        xx0 = 1. * x[i]
        f0 = f(x.reshape(N,))
        x[i] = x[i] + eps
        f1 = f(x.reshape(N,))
        val = np.array([f1 - f0]).item()
        gradient.append(val/eps)
        x[i] = xx0
    return np.array(gradient).reshape(x.shape)


def hessian(x, f):
    """Computes the Hessian matrix of f at point x (using the Jacobian of the gradient)."""
    N = x.shape[0]
    hessian = np.zeros((N,N)) 
    gd_0 = gradient_f(x, f)
    eps = np.linalg.norm(gd_0) * np.finfo(np.float32).eps 
    for i in range(N):
        xx0 = 1.*x[i]
        x[i] = xx0 + eps
        gd_1 =  gradient_f(x, f)
        hessian[:, i] = ((gd_1 - gd_0)/eps).reshape(x.shape[0])
        x[i] = xx0
    return hessian