import numpy as np
import sequence_jacobian as ssj
from numba import njit

##############################################################################################################################################
#  1. Heterogeneous households
##############################################################################################################################################

##### Household functions #####

def utility(c, n, eis, frisch, vphi):
    """Computes the individual utilities"""
    σ  = 1/eis
    ν  = 1/frisch
    if σ == 1:
        util = np.log(c)
    else:
        util = ((c**(1-σ))/(1-σ))
    if ν == 1:
        util -= vphi * np.log(n)
    else:
        util -= vphi * ((n**(1+ν))/(1+ν))
    return util


def value_function(util, a, a_grid, Pi, beta, tol=1e-8, max_iter=10_000, verbose=False, v_init=None):
    """Computes the individual value function values"""
    a_i, a_pi = get_lottery(a, a_grid)
    if v_init is None:
        v = -np.ones_like(a) # Initial guess
    err = np.inf
    i   = 0
    while err > tol and i < max_iter:
        v_old = v
        v     = util + beta * expectation_iteration(v, Pi, a_i, a_pi)
        err   = np.max(np.abs(v - v_old))
        i    += 1
        if verbose:
            print('Iteration {}: error = {}'.format(i, err))
    return v


def get_lottery(a, a_grid):
    """For any policy a, we can identify the points a_i and a_{i+1} in the grid it is closest to
    as well as the weights (pi, 1-pi) associated.
    Returns index in a_grid a_i and the weight associated a_pi."""
    # step 1: find the i such that a' lies between gridpoints a_i and a_(i+1)
    a_i = np.searchsorted(a_grid, a) - 1
    
    # step 2: obtain lottery probabilities pi
    a_pi = (a_grid[a_i+1] - a)/(a_grid[a_i+1] - a_grid[a_i])
    return a_i, a_pi


def expectation_iteration(X, Pi, a_i, a_pi):
    Xend = Pi @ X
    return expectation_policy(Xend, a_i, a_pi)


@njit
def expectation_policy(Xend, a_i, a_pi):
    X = np.zeros_like(Xend)
    for e in range(a_i.shape[0]):
        for a in range(a_i.shape[1]):
            # expectation is pi(e,a)*Xend(e,i(e,a)) + (1-pi(e,a))*Xend(e,i(e,a)+1)
            X[e, a] = a_pi[e, a]*Xend[e, a_i[e, a]] + (1-a_pi[e, a])*Xend[e, a_i[e, a]+1]   
    return X


def lintrans(pente):
    """"Function to define linear transfer schedule."""
    x = np.linspace(0, 6, 7)
    t_rule = -pente*x+pente*x[3]
    return t_rule 


def transfers(pi_e, Div, Tax, Trans, Subv, Subv1, Subv2, e_grid):    
    """Transfers to households."""
    tax_rule = e_grid
    tax   = Tax / np.sum(pi_e * tax_rule) * tax_rule

    # benchmark
    trans_rule = np.linspace(1.5, 0.5, 7) 
    trans = Trans / np.sum(pi_e * trans_rule) * trans_rule
    
    div_rule = e_grid
    div   = Div / np.sum(pi_e * div_rule) * div_rule
    
    subv_rule  = np.ones_like(e_grid)
    subv  = Subv / np.sum(pi_e * subv_rule) * subv_rule
    
    subv1_rule = lintrans(1.25)
    subv1 = Subv1 / np.sum(pi_e * subv_rule) * subv1_rule
    
    subv2_rule = np.ones_like(e_grid)
    subv2_rule[3:] = 0
    subv2 = Subv2 / np.sum(pi_e * subv_rule) * subv2_rule
    
    T = div + trans + subv + subv1 + subv2 - tax
    return T


def household_init(a_grid, e_grid, w, r, taul, T, tauc, L, pFE_HA, cbar, eis):
    """Initialisation of the household's block."""
    ws = w * e_grid
    coh = (1 + r) * a_grid[np.newaxis, :] + (1-taul) * ws[:, np.newaxis] * L + T[:, np.newaxis] - pFE_HA * (1+tauc) * cbar
    Va  = ((1 + r)/(1+tauc)) * (0.1 * coh) ** (-1 / eis) 
    return Va


def welfare(c, a, n, eis, frisch, vphi, a_grid, Pi, beta):
    # Compute utility and value function
    util = utility(c, n, eis, frisch, vphi)
    v = value_function(util, a, a_grid, Pi, beta) # Value function at the steady-state
    
    for prod in range(c.shape[0]):
        # Utility
        globals()['util_%s' % prod] = np.zeros_like(util)
        globals()['util_%s' % prod][prod, :] = util[prod, :]
        # Value function
        globals()['v_%s' % prod] = np.zeros_like(v)
        globals()['v_%s' % prod][prod, :] = v[prod, :]
        
    return util, v, util_0, util_1, util_2, util_3, util_4, util_5, util_6, v_0, v_1, v_2, v_3, v_4, v_5, v_6


def productivity_breakdown(e_grid, c, cFE, cH, c_tot, a, n, ns, inc):
    # per productivity levels
    for prod in range(len(e_grid)):
        # Consumption
        globals()['c_%s' % prod] = np.zeros_like(c)
        globals()['c_%s' % prod][prod, :] = c[prod, :]
        # Consumption of energy
        globals()['cFE_%s' % prod] = np.zeros_like(cFE)
        globals()['cFE_%s' % prod][prod, :] = cFE[prod, :]
        # Consumption of home good
        globals()['cH_%s' % prod] = np.zeros_like(cH)
        globals()['cH_%s' % prod][prod, :] = cH[prod, :]
        # Consumption
        globals()['ctot_%s' % prod] = np.zeros_like(c_tot)
        globals()['ctot_%s' % prod][prod, :] = c_tot[prod, :]
        # Assets
        globals()['a_%s' % prod] = np.zeros_like(a)
        globals()['a_%s' % prod][prod, :] = a[prod, :]
        # Labour
        globals()['n_%s' % prod] = np.zeros_like(n)
        globals()['n_%s' % prod][prod, :] = n[prod, :]
        # Effective labour
        globals()['ns_%s' % prod] = np.zeros_like(ns)
        globals()['ns_%s' % prod][prod, :] = ns[prod, :]
        # Total net income
        globals()['inc_%s' % prod] = np.zeros_like(inc)
        globals()['inc_%s' % prod][prod, :] = inc[prod, :]
        
    return  c_0, c_1, c_2, c_3, c_4, c_5, c_6, ctot_0, ctot_1, ctot_2, ctot_3, ctot_4, ctot_5, ctot_6, a_0, a_1, a_2, a_3, a_4, a_5, a_6, n_0, n_1, n_2, n_3, n_4, n_5, n_6, ns_0, ns_1, ns_2, ns_3, ns_4, ns_5, ns_6, inc_0, inc_1, inc_2, inc_3, inc_4, inc_5, inc_6, cFE_0, cFE_1, cFE_2, cFE_3, cFE_4, cFE_5, cFE_6, cH_0, cH_1, cH_2, cH_3, cH_4, cH_5, cH_6


def make_grids(rho_s, sigma_s, nS, amin, amax, nA):
    a_grid = ssj.grids.agrid(amin=amin, amax=amax, n=nA)
    e_grid, pi_e, Pi = ssj.grids.markov_rouwenhorst(rho=rho_s, sigma=sigma_s, N=nS)
    return a_grid, e_grid, pi_e, Pi


##### Household block #####

@ssj.het(exogenous='Pi', policy='a', backward='Va', backward_init=household_init)
def household(Va_p, a_grid, e_grid, 
              beta, eis, cbar, 
              pFE_HA, w, r, taul, T, tauc, L, YFE, YF):
    """Households' program."""
    # Real wage weighted by productivity
    ws = w * e_grid

    # uc_t(a_t)
    uc_nextgrid = (1+tauc) * beta * Va_p
    
    # c(z_t, a_t)
    c_nextgrid = uc_nextgrid ** (-eis)
    
    # c(z_t, a_{t-1})
    coh = (1 + r) * a_grid[np.newaxis, :] + (1-taul) * ws[:, np.newaxis] * L + T[:, np.newaxis] - pFE_HA * (1+tauc) * cbar
    a = ssj.utilities.interpolate.interpolate_y((1 + tauc)*c_nextgrid + a_grid, coh, a_grid)
    ssj.misc.setmin(a, a_grid[0])
    c = (coh - a) / (1 + tauc)
   
    # calculate marginal utility to go backward
    Va = ((1 + r)/(1+tauc)) * c ** (-1 / eis)
    
    c_tot = c + pFE_HA * cbar
    n     = L * np.ones_like(c)
    ns    = e_grid[:, np.newaxis] * n
    inc   = r * a_grid[np.newaxis, :] + w * (1-taul) * ns + T[:, np.newaxis]
    
    cH  = ((YF-YFE)/YF) * c
    cFE = (YFE/YF) * c + cbar
    
    return Va, a, c, c_tot, n, ns, inc, cFE, cH


# Definition of the household block
household = household.add_hetinputs([transfers, make_grids])
household = household.add_hetoutputs([productivity_breakdown]) # welfare



##############################################################################################################################################
#  2. Steady state
##############################################################################################################################################

##### Blocks of equations for the steady state #####

@ssj.simple
def equations_ss(share, r, B_b, B_g, B_t, L, mu, se, sce, alpphaE, etaE, tauc, sF):
    alpphaf = alpphaE * (1/share - 1) / (1-alpphaE)
    Z  = 1
    i  = r
    Br = B_b
    mc = 1/mu
    mcF  = mc
    pF   = mcF
    pFE  = (1/Z) * ( ((se)/(alpphaE + alpphaf*(1-alpphaE))) * (mc**(-etaE)) )**(1/(1-etaE))
    pFE_HA = pFE
    pH   = ( (1/(1-alpphaE)) * ( (mcF**(1-etaE)) - (alpphaE * (pFE)**(1-etaE)) ) )**(1/(1-etaE))
    mcH  = pH
    w    = ( (1/(1-alpphaf)) * ( ((Z*mcH)**(1-etaE)) - (alpphaf * (pFE)**(1-etaE)) ) )**(1/(1-etaE))
    YH   = (L / (1-alpphaf)) * (Z**(1-etaE)) * (w/mcH)**(etaE)
    E    = alpphaf * YH * (pFE/mcH)**(-etaE) * Z**(etaE-1)
    YF   = (YH / (1-alpphaE)) * (pH/mcF)**etaE
    Y    = YF
    YFE  = alpphaE * YF * (pFE/mcF)**(-etaE)
    cbar = sce * Y / pFE
    incomp = pFE * cbar
    Esupply = se * Y / pFE
    Div  = Y * (1 - mc)
    B    = B_b * Y 
    G    = B_g * Y
    Trans= B_t * Y
    Subv = 0
    Subv1 = 0
    Subv2 = 0
    pi = 0
    piw = 0
    piH = 0
    piF = 0
    piFE = 0
    subv = 0
    bouclier = (pFE - pFE_HA) * YFE + (pFE - pFE_HA) * tauc * cbar + sF * pFE * E
    bouclier_Y = bouclier / Y
    bouclier_Yss = bouclier / Y
    Dep  = G + Subv + Subv1 + Subv2 + bouclier
    TransEXO = 0
    TaxEXO   = 0
    adj_p = 0
    i_shock = 0
    return alpphaf, Z, i, Br, mc, mcF, pF, pFE, pFE_HA, pH, mcH, w, YH, E, YF, Y, YFE, cbar, incomp, Esupply, Div, B, G, Trans, Subv, Subv1, Subv2, pi, piw, piH, piF, piFE, subv, bouclier, bouclier_Y, bouclier_Yss, Dep, TransEXO, TaxEXO, adj_p, i_shock


@ssj.simple
def definitions_ss(pFE, w, L, C, cbar, Tax, tauc, taul):
    Recettes     = Tax + taul * w * L + tauc * C + tauc * pFE * cbar
    Capprox      = C
    conso        = C - Capprox
    i_res        = 0
    debt_res     = 0
    nkpc_res     = 0
    tauc_res     = 0
    taul_res     = 0
    return Recettes, Capprox, conso, i_res, debt_res, nkpc_res, tauc_res, taul_res


@ssj.simple
def market_clearing_ss(A, B, Esupply, cbar, YFE, E, vphi, w, taul, tauc, C, L, eis, frisch, muw, Y, G, pFE, Tax, r, B_b, B_g, B_t):
    asset_mkt    = A - B
    energy_mkt   = Esupply - YFE - E
    labour_mkt   = vphi - w * ((1 - taul)/(1 + tauc)) * (C**(-1/eis)) * (L**(-1/frisch)) / muw
    homegood_mkt = Y - (C + pFE*cbar + G + pFE * Esupply)
    balanced_bud = taul * w * L + tauc * (C + pFE * cbar) + Tax - (r * B_b + B_g + B_t) * Y
    return asset_mkt, energy_mkt, labour_mkt, homegood_mkt, balanced_bud


##### Definition of the steady state model #####

model_ss = ssj.create_model(blocks=[household, equations_ss, definitions_ss, market_clearing_ss], name='CepreHANK steady-state')



##############################################################################################################################################
#  3. Dynamics
##############################################################################################################################################

##### Blocks of equations for the dynamics #####

@ssj.simple
def firm(Y, w, pi, alpphaE, alpphaf, etaE, mu, kappa, sF, Z, G, pFE, pFE_HA):   
    pH  = (1/Z) * ( alpphaf*((1-sF)*pFE)**(1-etaE) + (1-alpphaf)*w**(1-etaE) )**(1/(1-etaE))
    pF  = ( alpphaE*(pFE_HA)**(1-etaE) + (1-alpphaE)*pH**(1-etaE) )**(1/(1-etaE))
    mcF = pF
    mcH = pH
    
    YH  = (1-alpphaE) * ( (pH/mcF)**(-etaE) ) * Y 
    YFE = alpphaE * ( (pFE_HA/mcF)**(-etaE) ) * Y
    L   = (1-alpphaf) * ( (w/mcH)**(-etaE) ) * YH * Z**(etaE-1)
    E   = alpphaf * ( ((1-sF)*pFE/mcH)**(-etaE) ) * YH * Z**(etaE-1)
    
    adj_p = Y * (((mu/(mu-1))/kappa)/2)*(pi**2)
    Div   = Y - pF * Y - adj_p
    
    return pH, pF, YH, E, L, YFE, Div, adj_p, mcF, mcH


@ssj.simple
def subsidy(pFE, subv):
    pFE_HA = pFE - subv
    return pFE_HA


@ssj.simple
def energy_policy(pFE, pFE_HA, G, Subv, Subv1, Subv2, YFE, E, cbar, tauc, sF):
    bouclier = (pFE - pFE_HA) * YFE + (pFE - pFE_HA) * tauc * cbar + sF * pFE * E
    Dep   = bouclier + G + Subv + Subv1 + Subv2
    return bouclier, Dep


@ssj.simple
def mkt_clearing(A, B, C, Capprox, G, Y, L, w, pFE, Esupply, adj_p, χc, χl, bouclier, cbar, tauc, taul):
    asset_mkt  = A - B
    homegood_mkt = Y - C - pFE * cbar - pFE * Esupply - G - adj_p
    conso      = Capprox - C
    tauc_res   = (-tauc + tauc.ss) * (C + pFE * cbar) + (χc * bouclier)
    taul_res   = (-taul + taul.ss) * (w * L) + (χl * bouclier)
    return asset_mkt, homegood_mkt, conso, tauc_res, taul_res


@ssj.simple
def inflation(pF, pH, pFE):
    piF = pF / pF(-1) - 1
    piH = pH / pH(-1) - 1
    piFE = pFE / pFE(-1) - 1
    return piF, piH, piFE


@ssj.simple
def nkpc(pi, pF, Y, r, mu, kappa):
    nkpc_res = -pi + kappa * (pF  - 1/mu)  + (1/(1+r(+1))) * (Y(+1)/Y) * pi(+1)
    return nkpc_res


@ssj.simple
def union(C, L, w, eis, vphi, frisch, muw, kappaw, taul, tauc, beta, piw):
    uprime      = C**(-1/eis)
    vprime      = vphi * (L**(1/frisch))
    parenthesis = L * vprime - (1/muw) * ((1 - taul)/(1 + tauc)) * w * L * uprime
    labour_mkt   = kappaw * parenthesis + beta * piw(+1) - piw
    return labour_mkt


@ssj.simple
def wage(pi, w):
    piw = (1 + pi) * w / w(-1) - 1
    return piw


@ssj.simple
def definitions(B, Y, YFE, E, pFE, cbar, bouclier):
    Br         = B / Y
    incomp     = pFE * cbar
    Esupply    = YFE + E 
    bouclier_Y = bouclier / Y
    bouclier_Yss = bouclier / Y.ss
    return Br, incomp, Esupply, bouclier_Y, bouclier_Yss


@ssj.simple
def mkt_clearing(A, B, C, Capprox, G, Y, L, w, pFE, Esupply, adj_p, χc, χl, bouclier, cbar, tauc, taul):
    asset_mkt  = A - B
    homegood_mkt = Y - C - pFE * cbar - pFE * Esupply - G - adj_p
    conso      = Capprox - C
    tauc_res   = (-tauc + tauc.ss) * (C + pFE * cbar) + (χc * bouclier)
    taul_res   = (-taul + taul.ss) * (w * L) + (χl * bouclier)
    return asset_mkt, homegood_mkt, conso, tauc_res, taul_res


@ssj.simple
def estim_eq(pFE, TransEXO, G):
    pFE_out = pFE
    TransEXO_out = TransEXO
    G_out = G
    return pFE_out, TransEXO_out, G_out


# When tax rates are fixed at their ss values
@ssj.simple
def mkt_clearing_exo(A, B, C, Capprox, G, Y, pFE, Esupply, adj_p, cbar):
    asset_mkt  = A - B
    homegood_mkt = Y - C - pFE * cbar - pFE * Esupply - G - adj_p
    conso      = Capprox - C
    return asset_mkt, homegood_mkt, conso