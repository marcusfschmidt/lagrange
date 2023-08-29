#%%
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
ti = 0
tf = 20
t = np.linspace(ti, tf, 1500)



# Use style file for plots
plt.style.use('../plotstyle.mplstyle')
# Import plotting function to ensure consistency between plots
import sys; sys.path.append(".."); from figSetup import figSetup

### Define constants; only some of them are used depending on the system
g = 9.81 # Gravitational acceleration
# m = 1 # Mass of particles
l = 1 # Length of pendulums
dknArray = np.linspace(0.1, 4.5, 50)
dknArray = np.array([2.9, 3, 3.1])
dkArray = np.zeros_like(dknArray)

# dkArray = np.array([0])
# dkArray = np.linspace(-1, 1, 50)
gam = 11.03625

Es = 3.67875
Ep = -14.6966

q1M = np.zeros((len(dkArray), len(t)))
q2M = np.zeros((len(dkArray), len(t)))

def equations2(vars, *args):
    x, theta = vars
    k, g, l, m = args

    om0 = np.sqrt(g/(l + m*g/k)) # Natural frequency of the first pendulum
    oms = np.sqrt(k/m) # Natural frequencies of the system

    # eq1 = m*(oms**2 - om0**2)/(2*l) - gam
    eq3 = potentialSpring([x, theta], [0, 0], 0) - Es
    eq4 = potentialPendulum([x, theta], [0, 0], 0) - Ep
    return (eq3, eq4)


def equations(vars, *args):
    k = vars
    dk, gam, m, g = args

    om0 = np.sqrt(g/(l + m*g/k)) # Natural frequency of the first pendulum
    oms = np.sqrt(k/m) # Natural frequencies of the system

    eq1 = m*(oms**2 - om0**2)/(2*l) - gam
    eq2 = 2*om0 - oms - dk
    return (eq1, eq2)

from scipy.optimize import fsolve, least_squares

from numpy import sqrt as sqrt
from numpy import arccos as acos


m = 1
for n, dkn in enumerate(dknArray):
    k = g*m*dkn/l
    # m = 1
    #solve equations using fsolve
    # m, k =  fsolve(equations, (1, 29.43), args = (dk, gam, l, g), maxfev = 10000)
    # k =  fsolve(equations2, (20), args = (dk, g, l, m), maxfev = 10000)[0]
    # sol = least_squares(equations2, (1, 29.43, 0.5, 0.05), args = (dk, gam, l, g, Es, Ep), bounds = ([0, 0, -l, -np.inf], [np.inf, np.inf, np.inf, np.inf]))
    # m, k, x, theta = sol.x

    # x, theta = fsolve(equations2, (0.5, 0.05), args = (k, g, l, m), maxfev = 10000)
    # sol = least_squares(equations2, (0.5, 0.05), args = (k, g, l, m), bounds = ([0, -np.inf], [np.inf, np.inf]))
    # x, theta = sol.x
    # print(x, theta)
    x = sqrt(2)*sqrt(Es/k)
    theta = -acos(-Ep/(g*m*(l + sqrt(2)*sqrt(Es/k))))
    print(Ep/(g*m*(l + sqrt(2)*sqrt(Es/k))))
    print(x, theta)

    #%%

    om0 = np.sqrt(g/(l + m*g/k)) # Natural frequency of the first pendulum
    oms = np.sqrt(k/m) # Natural frequencies of the system

    dk = 2*om0 - oms
    dkArray[n] = dk 
    # gam = m*(oms**2 - om0**2)/(2*l)

    ### Friction coefficients when including Rayleigh disspiation
    # Friction coefficient ~ v^0, e.g. friction from the normal force of a surface
    kineticFriction = np.array([0.1*m*g, 0.1*m*g])*0
    # Define the static friction coefficient when v ~ 0
    staticFriction = np.array([0.2*m*g, 0.2*m*g])*0
    # Flag for boolean flipping the sign of the friction coefficient when velocity changes sign
    flipSignBoolean = True
    # Friction coefficient ~ v^1
    linearFrictionCoefficient = np.array([0.1, 0.1])*0
    # If higher order friction is needed, the dissipation function must be changed accordingly

    # x = 0.5
    # theta = 0.05

    ### Initial conditions for the generalized coordinates and their deriatives 
    q0 = np.array([x, theta])
    v0 = np.array([0, 0])
    y0 = np.concatenate((q0, v0))

    # print(kineticSpring(q0, v0, 0) + potentialSpring(q0, v0, 0))
    # print(kineticPendulum(q0, v0, 0) + potentialPendulum(q0, v0, 0))

    # ### Define the time array 
    # ti = 0
    # tf = 60
    # t = np.linspace(ti, tf, 1000)

    ### Define step size for numerical differentiation
    h = 1e-4
    ### Define the tolerances of the solver
    rtol = 1e-7
    atol = 1e-7


    import numpy as np
    import matplotlib.pyplot as plt

    def kineticSpring(q, v, t):
        q1, q2 = q
        v1, v2 = v


        return 0.5 * m*v1**2
    
    def kineticPendulum(q, v, t):
        q1, q2 = q
        v1, v2 = v


        return 0.5 * m*(l + q1)**2*v2**2
    
    def potentialSpring(q, v, t):
        q1, q2 = q


        return 0.5*k*q1**2
    
    def potentialPendulum(q, v, t):
        q1, q2 = q


        return -g * m *(l + q1)*np.cos(q2)

    ### Define the kinetic and potential energy functions
    def kinetic(q, v, t):
        q1, q2 = q
        v1, v2 = v
  

        return kineticPendulum(q, v, t) + kineticSpring(q, v, t)
        # return 0.5 * m1 *(v1**2 + (l1 + q1)**2*v2**2)
        # # Kinetic energy terms for a double pendulum
        # T1 = 0.5 * m1 * l1**2 * v1**2
        # T2 = 0.5 * m2 * (l1**2 * v1**2 + l2**2 * v2**2 + 2 * l1 * l2 * v1 * v2 * np.cos(q1 - q2))
        # return T1 + T2

    # Potential energy function
    def potential(q, v, t):
        q1, q2 = q
        v1, v2 = v 

        return potentialPendulum(q, v, t) + potentialSpring(q, v, t)
        # return -g * m1 *(l1 + q1)*np.cos(q2) + 0.5*k1*q1**2

        # # Potential energy terms for a double pendulum
        # U1 = -m1 * g * l1 * np.cos(q1)
        # U2 = -m2 * g * (l1 * np.cos(q1) + l2 * np.cos(q2))
        # return U1 + U2

    ### Define the Lagrangian with. j is a dummy variable needed to perform mixed partial derivatives.
    # Unlike for the n = 1 script, the damping function exp(ct) is not included and must be incorporated via the dissipation function.
    def lagrangian(q, q_dot, t, j = 0):
        return kinetic(q, q_dot, t) - potential(q, q_dot, t)


    ### Define function to approximate the first order derivative of a function f(q, q_dot, t, j).
    def firstOrderDerivative(func, q, q_dot, t, i = 0, j = 0, H_q = 0, H_qdot = 0, H_t = 0):
        """ 
        func: function to be differentiated of the form f(q, q_dot, t, j)
        q: values of q
        q_dot: values of q_dot
        t: value of t
        i: index of q, q_dot or t to differentiate func with respect to
        j: index of q, q_dot or t to feed to func as i, used for mixed partial derivatives
        H_q: step size for q. Only one of H_q, H_qdot, H_t can be non-zero
        H_qdot: step size for q_dot. Only one of H_q, H_qdot, H_t can be non-zero
        H_t: step size for t. Only one of H_q, H_qdot, H_t can be non-zero
        
        """

        # Define and populate arrays for step sizes to ensure differentation with respect to the correct variable
        h_q = np.zeros(2)
        h_qdot = np.zeros(2)
        h_q[i] = H_q
        h_qdot[i] = H_qdot

        # Central difference approximation of the first derivative of O(h^2)
        return (func(q + h_q, q_dot + h_qdot, t + H_t, j = j) - func(q - h_q, q_dot - h_qdot, t - H_t, j = j)) / (2 * ( H_q + H_qdot + H_t))


    ### Define the dissipation function to be included in the Euler-Lagrange equation. j is a dummy variable needed to perform mixed partial derivatives.
    def dissipation(q, q_dot, t, j = 0):
        """
        Returns the dissipation function for a system with two generalized coordinates, neglecting particle-particle interactions (only diagonal terms in the dissipation matrix are non-zero).
        """
        q1_dot, q2_dot = q_dot

        # Zeroth order friction for mass 1 and 2
        D_01 = (np.sign(q1_dot) if flipSignBoolean else 1) * (staticFriction[0] if np.abs(q1_dot) < 1e-5 else kineticFriction[0])*q1_dot
        D_02 = (np.sign(q2_dot) if flipSignBoolean else 1) * (staticFriction[1] if np.abs(q2_dot) < 1e-5 else kineticFriction[1])*q2_dot

        # First order friction for mass 1 and 2
        D_11 = 1/2*linearFrictionCoefficient[0]*q1_dot**2
        D_12 = 1/2*linearFrictionCoefficient[1]*q2_dot**2

        return D_01 + D_02 + D_11 + D_12



    ### Define helper functions for repeated differentiation, with j passed as the i-variable to firstOrderDerivative
    def dL_dq(q, q_dot, t, j = 0):
        return firstOrderDerivative(lagrangian, q, q_dot, t, i = j, H_q = h)

    def dL_dqdot(q, q_dot, t, j = 0):
        return firstOrderDerivative(lagrangian, q, q_dot, t, i = j, H_qdot = h)

    ### Define the ODE to be solved. 
    def ODE(t, y):
        """
        Returns the ODE to be solved to extract the EOM for a system with two generalized coordinates.
        """
        q1, q2, q1_dot, q2_dot = y

        q = np.array([q1, q2])
        q_dot = np.array([q1_dot, q2_dot])


        # Euler-Lagrange equation for a system with two generalized coordinates. i and j indices are used to perform the mixed partial derivatives.
        
        # With each iteration, the derivatives of the Lagrangian are evaluated at the current values of q, q_dot and t.
        K1 = dL_dq(q, q_dot, t, j = 0) - firstOrderDerivative(dL_dqdot, q, q_dot, t, j = 0, H_t = h) - firstOrderDerivative(dissipation, q, q_dot, t, i = 0, H_qdot = h)
        K2 = dL_dq(q, q_dot, t, j = 1) - firstOrderDerivative(dL_dqdot, q, q_dot, t, j = 1, H_t = h) - firstOrderDerivative(dissipation, q, q_dot, t, i = 1, H_qdot = h)
        A1 = firstOrderDerivative(dL_dqdot, q, q_dot, t, i = 0, j = 0, H_q = h)
        B1 = firstOrderDerivative(dL_dqdot, q, q_dot, t, i = 0, j = 0, H_qdot = h)
        C1 = firstOrderDerivative(dL_dqdot, q, q_dot, t, i = 1, j = 0, H_q = h)
        D1 = firstOrderDerivative(dL_dqdot, q, q_dot, t, i = 1, j = 0, H_qdot = h)

        A2 = firstOrderDerivative(dL_dqdot, q, q_dot, t, i = 0, j = 1, H_q = h)
        B2 = firstOrderDerivative(dL_dqdot, q, q_dot, t, i = 0, j = 1, H_qdot = h)
        C2 = firstOrderDerivative(dL_dqdot, q, q_dot, t, i = 1, j = 1, H_q = h)
        D2 = firstOrderDerivative(dL_dqdot, q, q_dot, t, i = 1, j = 1, H_qdot = h)
        
        # The ODE is then solved for q_dot_dot:
        q1_dot_dot = ((A2*q1_dot + C2*q2_dot - K2)*D1 - D2*(A1*q1_dot + C1*q2_dot - K1))/(B1*D2 - B2*D1)
        q2_dot_dot = ((-A2*q1_dot - C2*q2_dot + K2)*B1 + B2*(A1*q1_dot + C1*q2_dot - K1))/(B1*D2 - B2*D1)

        return np.array([q1_dot, q2_dot, q1_dot_dot, q2_dot_dot])

    #use solve_ivp
    sol = solve_ivp(ODE, [0, tf], y0, t_eval=t, method='RK45', rtol=rtol, atol=atol)
    q1 = sol.y[0]
    q2 = sol.y[1]
    q1_dot = sol.y[2]
    q2_dot = sol.y[3]

    q1M[n,:] = q1
    q2M[n,:] = q2/(potential(q0, v0, 0) + kinetic(q0, v0, 0))


    # q1M[n] = kineticSpring([q1, q2], [q1_dot, q2_dot], t) + potentialSpring([q1, q2], [q1_dot, q2_dot], t)
    # q2M[n] = kineticPendulum([q1, q2], [q1_dot, q2_dot], t) + potentialPendulum([q1, q2], [q1_dot, q2_dot], t)
    # initEnergy[N] = kinetic([q1, q2], [q1_dot, q2_dot], t) + potential([q1, q2], [q1_dot, q2_dot], t)




# # Calculate the energy at each time step
# totalEnergy = np.zeros_like(t)
# kineticEnergy = np.zeros_like(t)
# potentialEnergy = np.zeros_like(t)

# totalSpring = np.zeros_like(t)
# totalPendulum = np.zeros_like(t)

# for i in range(len(t)):
#     q = np.array([q1[i], q2[i]])
#     q_dot = np.array([q1_dot[i], q2_dot[i]])

#     kineticEnergy[i] = kinetic(q, q_dot, t[i])
#     potentialEnergy[i] = potential(q, q_dot, t[i])
#     totalEnergy[i] = kineticEnergy[i] + potentialEnergy[i]


# # Calculate the relative energy difference as a function of time
# dE = (totalEnergy - totalEnergy[0])/totalEnergy[0]

#%%
fig, ax = figSetup(1, 1)
ax.pcolormesh(t, dkArray, q2M**2)

#%%
fig, ax = figSetup(1, 1)
for i in range(len(dkArray)):
    ax.plot(t, q2M[i,:]**2, label = r'$q_2(t)$')


#%%
#make outer product of t and dkArray to get a 2D array
t2D, dkArray2D = np.meshgrid(t, dkArray)
test = t2D**2*np.sinc(1/np.pi*(dkArray2D*t2D)/2)**2
fig, ax = figSetup(1, 1)
ax.pcolormesh(t, dkArray, test)

#%%

idx = 795
fig, ax = figSetup(1, 1)
ax.plot(dkArray, q2M[:,idx]**2, label = r'$q_2(t)$')















#%%
# # Plot the results
# fig, ax = figSetup(1, 3, sharex = True)
# ax[0].plot(t, q1, label=r'$q_1(t)$')
# ax[0].plot(t, q2, label=r'$q_2(t)$')
# ax[0].set_title(r'$q(t)$')

# ax[1].plot(t, kineticEnergy, label=r'$T(t)$')
# ax[1].plot(t, potentialEnergy, label=r'$V(t)$')
# ax[1].plot(t, totalEnergy, label=r'$E(t)$')

# ax[1].set_title(r'$T(t)$, $V(t)$ and $E(t)$')
# ax[2].plot(t, dE, label=r'$\Delta E(t)$')
# ax[2].set_title(r'$\Delta E(t)$')

# ax[0].set_xlabel(r'$t$')
# ax[1].set_xlabel(r'$t$')

# ax[0].legend()
# ax[1].legend()

# #%% Make animation

# # Set up the figure and axis
# fig, ax = plt.subplots()
# totalL = np.sum(l)
# ax.set_xlim(-totalL*1.1, totalL*1.1)
# ax.set_ylim(-totalL*1.1, totalL*1.1)
# ax.set_xlabel(r'$x$ [m]')
# ax.set_ylabel(r'$y$ [m]')

# # Define the strings to be displayed in the upper left corner
# massString = r'$m_1 = ' + str(m[0]) + r'$ kg, $m_2 = ' + str(m[1]) + r'$ kg'
# lengthString = r'$l_1 = ' + str(l[0]) + r'$ m, $l_2 = ' + str(l[1]) + r'$ m'

# # Add the strings in the upper left corner in a box
# ax.text(0.05, 0.95, massString + '\n' + lengthString, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# # Plot the pendulum as a line
# line, = ax.plot([], [], 'o-', lw=2)

# # Define animation function
# def animate(i):
#     x1 = np.sin(q1[i])*l[0]
#     y1 = -np.cos(q1[i])*l[0]
#     x2 = x1 + np.sin(q2[i])*l[1]
#     y2 = y1 - np.cos(q2[i])*l[1]

#     line.set_data([0, x1, x2], [0, y1, y2])
#     return line,


# interval = np.ceil(tf*1000/len(t))
# # Animate with duration that matches the final time of the simulation
# ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=interval, blit=True, repeat=False)

# fps = len(t)/tf
# plt.show()
# # ani.save('double_pendulum.mp4', writer = 'ffmpeg', fps = fps, dpi = 300)
#%%

