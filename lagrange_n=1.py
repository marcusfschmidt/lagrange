#%%
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Use style file for plots
plt.style.use('plotstyle.mplstyle')
# Import plotting function to ensure consistency
from figureSetup import figSetup

### Define constants
m = 1
g = 9.81

### Define step size for numerical differentiation
h = 1e-6

### Define function to approximate the first order derivative of a function f(q, q_dot, t)
def firstOrderDerivative(func, q, q_dot, t, h_q = 0, h_qdot = 0, h_t = 0):
    """
    func: function to be differentiated of the form f(q, q_dot, t)
    q: value of q
    q_dot: value of q_dot
    t: value of t
    h_q: step size for q. Only one of h_q, h_qdot, h_t can be non-zero
    h_qdot: step size for q_dot. Only one of h_q, h_qdot, h_t can be non-zero
    h_t: step size for t. Only one of h_q, h_qdot, h_t can be non-zero
    """

    # Central difference approximation of the first derivative of O(h^2)
    return (func(q + h_q, q_dot + h_qdot, t + h_t) - func(q - h_q, q_dot - h_qdot, t - h_t)) / (2 * ( h_q + h_qdot + h_t))


### Define the kinetic and potential energy functions for a simple pendulum
def kinetic(q, q_dot, t):
    return 0.5*m*(q_dot**2)

def potential(q, q_dot, t):
        return m*g*(1 - np.cos(q))


### Define the Lagrangian function with an arbitrary time-dependent coefficient
def lagrangian(q, q_dot, t, c = 0):
    """ 
    Returns the Lagrangian (T-V)*exp(c*t), with the exponential corresponding to linear damping of the system.
    """
    return (kinetic(q, q_dot, t) - potential(q, q_dot, t))*np.exp(c*t)


#### Define helper functions for repeated differentiation
def dL_dq(q, q_dot, t):
    return firstOrderDerivative(lagrangian, q, q_dot, t, h_q = h)

def dL_dqdot(q, q_dot, t):
    return firstOrderDerivative(lagrangian, q, q_dot, t, h_qdot = h)


### Define the ODE to be solved. The ODE is of second-order and is rewritten as a system of two first-order ODEs.
def ODE(t,y):
    q, q_dot = y
    """ 
    Returns the ODE to be solved to extract the EOM for a system with a single generalized coordinate.
    """

    # For one generalized coordinate, the equation to be solved can be shown to be:
    # d/dt dL/dq_dot + q_dot d/dq dL/dq_dot + q_dot_dot d/dq_dot dL/dq_dot = dL/dq,
    # such that
    # q_dot_dot = (dL/dq - d/dt dL/dq_dot - q_dot d/dq dL/dq_dot)/(d/dq_dot dL/dq_dot)
    
    # With each iteration, the derivatives of the Lagrangian are calculated with respect to q, q_dot and t:
    dL_dqdot_dq = firstOrderDerivative(dL_dqdot, q, q_dot, t, h_q = h)
    dL_dqdot_dqdot = firstOrderDerivative(dL_dqdot, q, q_dot, t, h_qdot = h)
    dL_dqdot_dt = firstOrderDerivative(dL_dqdot, q, q_dot, t, h_t = h)

    # The ODE is then solved for q_dot_dot:
    q_dot_dot = (dL_dq(q, q_dot, t) -  q_dot*dL_dqdot_dq - dL_dqdot_dt)/(dL_dqdot_dqdot)

    return np.array([q_dot, q_dot_dot])

# Initial conditions for the generalized coordinate and its derivative
q0 = np.array([np.pi/2, 0])


# Define the time array 
ti = 0
tf = 5
t = np.linspace(ti, tf, 10000)

# Solve the IVP using solve_ivp
sol = solve_ivp(ODE, (ti, tf), q0, t_eval = t, method = 'RK23')
q = sol.y[0]
q_dot = sol.y[1]

# Plot the solutions
fig, ax = plt.subplots(1, 1, figsize = (8, 6))
ax.plot(t, q, label=r'$\theta(t)$')
ax.plot(t, q_dot, label=r'$\dot{\theta}(t)$')
ax.set_xlabel(r'$t$')

analytical = q0[0] * np.cos(np.sqrt(9.81) * t)
ax.plot(t, analytical, label=r'$\theta(t)$, SAA')


# Calculate the total energy
totalEnergy = np.zeros_like(t)
for i in range(len(t)):
    totalEnergy[i] = kinetic(q[i], q_dot[i], t[i]) + potential(q[i], q_dot[i], t[i])



plt.legend()