import numpy as np
from scipy.linalg import cho_solve

kernels ={
        "Gaussian": lambda x1, x2, l: np.exp(-0.5*np.square((x1-x2)/l)),
        "Exp": lambda x1, x2, l: np.exp(-0.5/l*np.fabs(x1-x2)),
        "Matern32": lambda x1, x2, l: (1+np.sqrt(3)*np.fabs(x1-x2)/l)\
                        *np.exp(-np.sqrt(3)*np.fabs(x1-x2)/l),
    }

def kernel(kernel_function, x1, x2):
    """
    calculates kernel matrix K (pairwise for inputs in x1 and x2)

    Parameters
    __________
    f: function
        Function with two inputs and real value output
    x1: np.ndarray
        array containing data points
    x2: np.ndarray
        array containing data points

    Returns
    _______
    K: np.ndarray
        Matrix of pairwise evaluation of elements in x1 and x2
    """
    K = np.array([[kernel_function(x1_, x2_) for x2_ in x2]
                     for x1_ in x1])
    if np.array_equal(x1, x2): K = K + 1e-10*np.eye(K.shape[0])
    return K


def stack_column_wise(A):
    """ convenient name for .T.reshape(-1) operation """
    return A.T.reshape(-1)


def restack_column_wise(vec, N):
    """ convenient name """
    return vec.reshape(-1, N).T


def get_mean_cov(kxx, ktt, x, y, x_plot, t_plot):
    """
    calculates mean and covariance (cholesky factor) of a gaussian
    random field of a particular form

    A GP in position and time. We observe data (x, y) that are
    fixed in time. We can now look at the evolutio of a posterior
    GP (given (x,y)) in time by sampling from the random field.
    In particular we train on (x, t, y_x, y_t) and make use of the
    fact that the data (x, t, y_x, y_t) are actually t-independent.

    Parameters
    __________
    kxx: function
        Kernel function for space
    ktt: function
        Kernel function for time
    x: np.ndarray
        space input locations
    y: np.ndarray
        output of the space component
    x_plot: np.ndarray
        space locations where the posterior should be evaluated
    t_plot: np.ndarray
        time locations where the posterior should be evaluated

    Returns
    _______
    mu: np.ndarray
        mean of posterior GP - it is time independent
    Ctt: np.ndarray
        cholesky decomposition of kernel matrix corresponding to time
        (posterior is equivalent to prior)
    Cxx_post: np.ndarray
        cholesky decomposition of posterior kernel matrix corresponding to space
    diag: np.ndarray
        variance of each data input x
    """
    gitter = lambda n: 1e-9*np.eye(n)
    Ctt = np.linalg.cholesky( kernel(ktt, t_plot, t_plot) 
                                + gitter(t_plot.shape[0]) )
    Cxx = np.linalg.cholesky(kernel(kxx, x, x) + gitter(x.shape[0]))
    Kxsx = np.array( kernel(kxx, x_plot, x) )
    Kxsxs = np.array( kernel(kxx, x_plot, x_plot) )

    Chi = cho_solve((Cxx, True), Kxsx.T).T

    mu = Chi.dot(y)

    # post = Kxsxs - Kxsx Kxx^{-1} Kxsx.T
    post = Kxsxs - np.matmul(Chi, Kxsx.T)
    Cxx_post = np.linalg.cholesky(post)

    diag = np.diag(post)

    # return mean, cho(time), cho(position), variance(position)
    return mu, Ctt, Cxx_post, diag


def get_posterior_samples(x, y, N_samples, N_plot, T_plot, kernel_l, kernel_std, kernel_name):
    """
    get samples from a Gaussian random field.
    
    Parameters
    __________
    x: np.ndarray
        space input locations
    y: np.ndarray
        output of the space component
    N_samples: int
        number of posterior samples
    N_plot: int
        number of points to plot in the space direction
    N_plot: int
        number of points to plot in the time direction
    kernel_l: float
        lengthscale of the spacial kernel kxx
    kernel_std: float
        standard deviation of the spacial kernel kxx
    kernel_name: str
        one of ["Gaussian", "Exp"]

    Returns
    _______
    x_plot: np.ndarray
        input positions of the posterior GP samples in space
    y_plots: np.ndarray
        contains `N_samples` samples from the full process (space and time)
    mu: np.ndarray
        mean of the posterior GP in space (it is time-independent)
    std: np.ndarray
        std of each point in x_plot (needed for plotting the uncertainty)
    """
    ktt = lambda time1, time2: kernels["Gaussian"](time1, time2, 1.)
    kxx_ = kernels[kernel_name]
    kxx = lambda space1, space2: kernel_std*kxx_(space1, space2, kernel_l)

    N = x.shape[0]
    x_plot = np.linspace(-1, 4, N_plot)
    t_plot = np.linspace(0, 100., T_plot)

    mu, Ctt, Cxx_post, diag = get_mean_cov(kxx, ktt, x, y, x_plot, t_plot)
    std = np.sqrt(ktt(0, 0))*np.sqrt(diag)

    cov_cho = np.kron(Ctt, Cxx_post)

    y_plots = np.zeros((N_samples, N_plot, T_plot))
    for i in range(N_samples):
        eps = np.random.normal(size=cov_cho.shape[0])
        y_plot = mu.reshape(-1,1) \
                    + restack_column_wise(cov_cho.dot(eps), N_plot)
        y_plots[i, :] = y_plot

    return x_plot, y_plots, mu, std

