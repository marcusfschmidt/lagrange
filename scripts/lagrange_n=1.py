#%%
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Use style file for plots
plt.style.use('../plotstyle.mplstyle')
# Import plotting function to ensure consistency between plots
import sys; sys.path.append(".."); from figSetup import figSetup

### Define constants; only some of them are used depending on the system
m = 1 # Mass of particle
g = 9.81 # Gravitational acceleration
l = 1 # Length of pendulum
k = 10 # Spring constant

### Damping constant when including simple, linear damping. Corresponds to Rayleigh dissipation of order 1, v ~ 1.
dampingConstant = 0

### Friction coefficients when including Rayleigh disspiation
# Friction coefficient ~ v^0, e.g. friction from the normal force of a surface
kineticFriction = 0.1*m*g
# Define the static friction coefficient when v ~ 0
staticFriction = 0.2*m*g
# Flag for boolean flipping the sign of the friction coefficient when velocity changes sign
flipSignBoolean = True
# Friction coefficient ~ v^1
gamma = 0.05
linearFrictionCoefficient = 2*m*gamma
# If higher order friction is needed, the dissipation function must be changed accordingly


### Define step size for numerical differentiation
h = 1e-5

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


### Define the kinetic and potential energy functions 
def kinetic(q, q_dot, t):
    # Simple kinetic energy term
    return 0.5*m*q_dot**2

    # Kinetic energy term for a pendulum of length l
    # return 0.5*m*(l**2*q_dot**2)

def potential(q, q_dot, t):
        # Simple potential energy term
        # return m*g*q

        # Potential energy term for a pendulum of length l
        # return m*g*l*(1 - np.cos(q))

        # Potential energy term for a spring with spring constant k
        return 0.5*k*q**2


### Define the Lagrangian function with an arbitrary time-dependent coefficient
def lagrangian(q, q_dot, t):
    """ 
    Returns the Lagrangian (T-V)*exp(c*t), with the exponential corresponding to linear damping of the system.
    """
    return (kinetic(q, q_dot, t) - potential(q, q_dot, t))*np.exp(dampingConstant*t)

### Define the dissipation function to be included in the Euler-Lagrange equation
def dissipation(q, q_dot, t):
    """
    Returns the dissipation function for a system with a single generalized coordinate, neglecting particle-particle interactions.
    """
    # Dissipation function for a system with a single generalized coordinate, zeroth order friction
    D_0 =  (np.sign(q_dot) if flipSignBoolean else 1) * (staticFriction if np.abs(q_dot) < 1e-5 else kineticFriction)*q_dot

    # Dissipation function for a system with a single generalized coordinate, first order friction
    D_1 = 1/2*linearFrictionCoefficient*q_dot**2

    return D_0 + D_1


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
    # d/dt dL/dq_dot + q_dot d/dq dL/dq_dot + q_dot_dot d/dq_dot dL/dq_dot = dL/dq - dD/dq_dot,
    # such that
    # q_dot_dot = (dL/dq - dD/dq_dot - d/dt dL/dq_dot - q_dot d/dq dL/dq_dot)/(d/dq_dot dL/dq_dot)
    
    # With each iteration, the derivatives of the Lagrangian are calculated with respect to q, q_dot and t:
    dD_dqdot = firstOrderDerivative(dissipation, q, q_dot, t, h_qdot = h)
    dL_dqdot_dq = firstOrderDerivative(dL_dqdot, q, q_dot, t, h_q = h)
    dL_dqdot_dqdot = firstOrderDerivative(dL_dqdot, q, q_dot, t, h_qdot = h)
    dL_dqdot_dt = firstOrderDerivative(dL_dqdot, q, q_dot, t, h_t = h)
    
    # The ODE is then solved for q_dot_dot:
    q_dot_dot = (dL_dq(q, q_dot, t) - dD_dqdot - q_dot*dL_dqdot_dq - dL_dqdot_dt)/(dL_dqdot_dqdot)

    return np.array([q_dot, q_dot_dot])

# Initial conditions for the generalized coordinate and its derivative
y0 = np.array([1, 0.2])


# Define the time array 
ti = 0
tf = 5
t = np.linspace(ti, tf, 10000)

# Solve the IVP using solve_ivp
sol = solve_ivp(ODE, (ti, tf), y0, t_eval = t, method = 'RK23')
q = sol.y[0]
q_dot = sol.y[1]


######## Define the analytical small-angle approximation solution for a pendulum
# analytical = q0[0]*np.cos(np.sqrt(g/l)*t)


######## Define the analytical solution for an underdamped harmonic oscillator
######
# https://beltoforion.de/en/harmonic_oscillator/
# om0 = np.sqrt(k/m)
# om = np.sqrt(om0**2 - gamma**2)
# A = - np.sqrt(2*gamma*v0*x0 + v0**2 + x0**2*(gamma**2 + om**2))/(2*om)
# phi = -2*np.arctan((v0+gamma*x0)/(x0*om - np.sqrt(2*gamma*v0*x0+v0**2+x0**2*(gamma**2+om**2))))
# ######
# analytical = np.exp(-gamma*t)*2*A*np.cos(om*t + phi)


######## Define the analytical solution for a block moving along the horizontal axis with friction
analytical = -1/2*g*t**2 + y0[1]*t + y0[0]

# Calculate the energy at each time step
totalEnergy = np.zeros_like(t)
kineticEnergy = np.zeros_like(t)
potentialEnergy = np.zeros_like(t)

for i in range(len(t)):
    kineticEnergy[i] = kinetic(q[i], q_dot[i], t[i])
    potentialEnergy[i] = potential(q[i], q_dot[i], t[i])
    totalEnergy[i] = kineticEnergy[i] + potentialEnergy[i]

# Calculate the relative energy difference as a function of time
dE = (totalEnergy - totalEnergy[0])/totalEnergy[0]


# Plot the results
fig, ax = figSetup(1, 3, sharex = True)
ax[0].plot(t, q, label=r'$q(t)$')
ax[0].plot(t, q_dot, label=r'$\dot{q}(t)$')
# ax[0].plot(t, analytical, label=r'Analytical')
ax[0].set_title(r'$q(t)$ and $\dot{q}(t)$')

ax[1].plot(t, kineticEnergy, label=r'$T(t)$')
ax[1].plot(t, potentialEnergy, label=r'$V(t)$')
ax[1].plot(t, totalEnergy, label=r'$E(t)$')

ax[1].set_title(r'$T(t)$, $V(t)$ and $E(t)$')
ax[2].plot(t, dE, label=r'$\Delta E(t)$')
ax[2].set_title(r'$\Delta E(t)$')

ax[0].set_xlabel(r'$t$')
ax[1].set_xlabel(r'$t$')

ax[0].legend()
ax[1].legend()


