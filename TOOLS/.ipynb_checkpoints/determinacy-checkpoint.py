import numpy as np
from scipy.linalg import toeplitz
from numpy.fft import fftn, rfftn
from numba import njit
import matplotlib.pyplot as plt
import sequence_jacobian as ssj


def get_Jts(J, t, s, T=500):
    """Returns the Jacobian J of all variables at time t following all shocks at s
    
    Inputs
    -----------------
    J                : JacobianDict, Jacobian of all inputs (m) and outputs (n)
    t                : int, time of the effect we look at
    s                : int, time of the shocks considered
    T                : int, horizon of the Jacobian
    
    Outputs
    -----------------
    Jts              : array(n,m), impact of shocks in s on variables at t"""
    n = len(J.outputs)
    m = len(J.inputs)
    Jts = np.zeros((n, m))
    
    for (i, out) in enumerate(J.outputs):
        for (j, inp) in enumerate(J.inputs):
            if inp not in J[out].keys():
                Jts[i, j] = 0
            elif type(J[out][inp]) == ssj.classes.sparse_jacobians.SimpleSparse:
                Jts[i, j] = J[out][inp].matrix(T=T)[t, s]
            else:
                Jts[i, j] = J[out][inp][t, s]
            
    return Jts
    
    
def get_lastj(J, T=500, verbose=False):
    """Returns the symbol j of the Jacobian J (using the last column of the Jacobian)
    
    Inputs
    -----------------
    J                : JacobianDict, Jacobian of all inputs (m) and outputs (n)
    
    T                : int, horizon of the Jacobian
    
    Outputs
    -----------------
    j                 : array(2T-1,n,m), symbol of the Jacobian J"""
    n = len(J.outputs)
    m = len(J.inputs)
    j = np.zeros((2*T-1, n, m))
    
    # Negative terms
    for t in range(T):
        if verbose:
            print('Neg', t, 'Impact in {} of a shock in {}'.format(t, T-1))
        j[t] = get_Jts(J, t, T-1, T)
        
    # Positive terms
    for t in range(T-2, -1, -1):
        if verbose:
            print('Pos', 2*(T-1)-t, 'Impact in {} of a shock in {}'.format(T-1, t))
        j[2*(T-1)-t] = get_Jts(J, T-1, t, T)
    
    return j
    

def get_j(J, τ, T=500, verbose=False):
    """
    Returns the symbol j of the Jacobian J,
    a (2τ+1, n, m) object stacking the j_k (each of size n, m)
    
    Inputs
    ------------------
    J                 : array(n*T, m*T), Jacobian
    τ                 : int, horizon at which determinacy is evaluated
    T                 : int, horizon of the Jacobian

    Output
    ------------------
    A                 : array(T, n, m), stacked j_k associated with Jacobian J
    """
    n = len(J.outputs)
    m = len(J.inputs)
    assert τ <= T-1, "Determinacy can only be evaluated for τ smaller than the Jacobian's horizon"
    A = np.zeros((2*τ+1, n, m))
    
    A[τ, :, :] = get_Jts(J, τ, τ, T)
    for k in range(1, τ+1):
        # Negative k
        A[k-1, :, :] = get_Jts(J, k-1, τ, T)
        # Positive k
        A[τ+k, :, :] = get_Jts(J, τ, τ-k, T)
        if verbose:
            print('Neg', k-1, 'Impact in {} of a shock in {}'.format(k-1, τ))
            print('Pos', τ+k, 'Impact in {} of a shock in {}'.format(τ, τ-k))
    
    return A
    

def check_τ(j, verbose=False, tol=1e-8):
    """
    Check that τ is high enough (i.e. impact in 0 of a shock in τ is null and
    impact in τ of a shock at time 0 is null).
    
    Inputs
    ------------------
    j                 : array(2T-1,n,m), symbol of a Jacobian
    
    Output
    ------------------
    bool              : True if τ is high enough
    """
    if verbose:
        print('Maximum absolute impact in 0 of a shock in τ:', np.max(np.abs(j[0])))
        print('Maximum absolute impact in τ of a shock in 0:', np.max(np.abs(j[-1])))
    return np.isclose(np.max(np.abs(np.array([j[0], j[-1]]))), 0, atol=tol)


def check_E(j, jp, verbose=False, tol=1e-8):
    """
    Check that the correction term E is small enough.
    
    Inputs
    ------------------
    j                 : array(2T-1,n,m), symbol of a Jacobian
    jp                : array(2(T-1)-1,n,m), symbol of the previous period Jacobian
    
    Output
    ------------------
    bool              : True if E is small enough
    """
    comp = np.abs(j[1:-1, :, :] - jp)
    err  = np.max(comp)
    if verbose:
        print('The maximum error is ', err)
        print('It is reached for index ', np.unravel_index(np.argmax(comp), comp.shape))
    return err <= tol

    
def determinacy_auclert(J, plot=True, T=500, τ=None, tol=1e-5, zoom=None): 
    """Compute the winding number criterion for sequence-space models."""
    if τ is None:
        τ = T-1
    A  = get_j(J, τ, T)
    Ap = get_j(J, τ-1, T)
    
    if check_τ(A, tol=tol) and check_E(A, Ap, tol=tol):
        det_Aλ = detA_path(A)
        x, y   = det_Aλ.real, det_Aλ.imag
        
        if plot:
            if zoom is not None:
                if zoom[0] == 1:
                    x_selec = x[x >= zoom[1]]
                    y_selec = y[x >= zoom[1]]
                else:
                    x_selec = x[x <= zoom[1]]
                    y_selec = y[x <= zoom[1]]
                x = x_selec.copy()
                y = y_selec.copy()
            plt.plot(x, y, color='blue')
            plt.arrow(x[0], y[0], 0.001*(x[1]-x[-2]), 0.001*(y[1]-y[-2]), color='blue',
                  width=0.001, head_width=0.1, head_length=0.16)
            plt.plot(0, 0, marker='o', markersize=5, color='black')
            plt.axis('equal')
            plt.xlabel('Real axis')
            plt.ylabel('Imaginary axis')
            plt.show()
        return winding_number(x,y), x, y
        
        
        
#### Auclert et al's codes
def detA_path(A, N=4096):
    """Evaluates det A(lambda) at N equispaced points lambda on interval [0,2pi].

    A brief derivation of how this function uses FFT to rapidly evaluate det A(lambda) follows.

    We have, letting A_(-j) denote the k*k matrix A[-j,:,:]:

        det A(lambda) = det sum_(j=-(T-1))^(T-1) A_(-j)e^(i*j*lambda)
    
    which, flipping the order and realigning j, can be rewritten as

        e^(lambda*i*k*(T-1)) det sum_(j=0)^(2T-2) A_(-j+(T-1))e^(-i*j*lambda)   (***)

    Taking the sum in (***) for the values lambda=0,2*pi/N,...,2*pi*(N-1)/N, assuming N >= (2T-1),
    is just taking the discrete Fourier transform of the sequence A_(T-1),...,A_(-(T-1)),0,...,0
    right-padded with zeros to length N.
    
    Hence we can rapidly, simultaneously evaluate (***) at all points lambda equispaced from lambda=0
    to lambda=2*pi using the FFT. This is implemented below, with additional efficiency from fact that
    A(lambda) and A(2*pi-lambda) are conjugate.
    """
    # preliminary: assume and verify shape 2*T-1, k, k for A
    T = (A.shape[0]+1) // 2
    k = A.shape[1]
    if not (T == (A.shape[0]+1)/2 and N >= 2*T-1 and k == A.shape[2]):
        raise ValueError(f'Asymptotic A matrix has improper shape {A.shape}')

    # step 1: use FFT to calculate A(lambda) for each lambda = 2*pi*{0, 1/N, ..., 1/2} (last if N even)
    # note that we need to reverse order of A_t to get sequence A_(T-1),...,A_(-(T-1)),0,...,0
    Alambda = rfftn(A[::-1,...], axes=(0,), s=(N,))

    # step 2: take determinant of each, then multiply by e^(i*k*(T-1)*lambda) to get (***)
    det_Alambda = np.empty(N+1, dtype=np.complex128)
    det_Alambda[:N//2+1] = np.linalg.det(Alambda)*np.exp(2j*np.pi*k*(T-1)/N*np.arange(N//2+1))
    
    # step 3: use conjugate symmetry to fill in rest
    det_Alambda[N//2+1:] = det_Alambda[:(N+1)//2][::-1].conj()

    return det_Alambda


def winding_criterion(A, N=4096):
    """Build path of det A(lambda) and obtain its winding number, implementing winding number
    criterion for determinacy that generalizes Onatski (2006).

    Parameters
    ----------
    A : array ((2T-1)*k*k)
            asymptotic H_U matrix, where A[t,i,j] gives Jacobian of target i vs. unknown j
            at t-(T-1) above the main diagonal
    N : [optional] int
            number of equispaced points lambda on interval [0,2pi] for evaluating det A(lambda)

    Returns
    ----------
    winding_number : int
            winding number that characterizes existence and uniqueness of solutions:
                0 for determinate solution
                -1 (or lower) for indeterminacy 
                1 (or higher) for no solution
    """
    det_Alambda = detA_path(A, N)
    return winding_number(det_Alambda.real, det_Alambda.imag)


@njit
def winding_number(x, y):
    """Compute winding number around origin of (x,y) coordinates that make closed path by
    counting number of counterclockwise crossings of ray from (0,0) -> (infty,0) on x axis"""
    # ensure closed path!
    assert x[-1] == x[0] and y[-1] == y[0]

    winding_number = 0

    # we iterate through coordinates (x[i], y[i]), where cur_sign is flag for
    # whether current coordinate is above the x axis
    cur_sign = (y[0] >= 0)
    for i in range(1, len(x)):
        if (y[i] >= 0) != cur_sign:
            # if we're here, this means the x axis has been crossed
            # this generally happens rarely, so efficiency no biggie
            cur_sign = (y[i] >= 0)
            
            # crossing of x axis implies possible crossing of ray (0,0) -> (infty,0)
            # we will evaluate three possible cases to see if this is indeed the case
            if x[i] > 0 and x[i-1] > 0:
                # case 1: both (x[i-1],y[i-1]) and (x[i],y[i]) on right half-plane, definite crossing
                # increment winding number if counterclockwise (negative to positive y)
                # decrement winding number if clockwise (positive to negative y)
                winding_number += 2*cur_sign-1
            elif not (x[i] <= 0 and x[i-1] <= 0):
                # here we've ruled out case 2: both (x[i-1],y[i-1]) and (x[i],y[i]) in left 
                # half-plane, where there is definitely no crossing

                # thus we're in ambiguous case 3, where points (x[i-1],y[i-1]) and (x[i],y[i]) in
                # different half-planes: here we must analytically check whether we crossed
                # x-axis to the right or the left of the origin
                # [this step is intended to be rare]
                cross_coord = (x[i-1]*y[i] - x[i]*y[i-1])/(y[i]-y[i-1])
                if cross_coord > 0:
                    winding_number += 2*cur_sign-1
    return winding_number
    
    
def plot_jacobian_columns(J, T=500, shocks=[0, 100, 150, 200, 250], title=None):
    for s in shocks:
        if type(J) == ssj.classes.sparse_jacobians.SimpleSparse:
            plt.plot(J.matrix(T)[:, s], label='t={}'.format(s))
        else:
            plt.plot(J[:, s], label='t={}'.format(s))
    if title is not None:
        plt.title(title)
    plt.axhline(0, color='black', linestyle='--')
    plt.legend(frameon=False)
    plt.show()