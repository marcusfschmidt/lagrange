#%%
import sympy as sp
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#%%
# Define the variables and symbols
t = sp.symbols('t')
x, y = sp.symbols('x y', cls=sp.Function)

# Define the differential equations
dx_dt = x(t).diff(t)
dy_dt = y(t).diff(t)
d2x_dt2 = x(t).diff(t, t)
d2y_dt2 = y(t).diff(t, t)

eq1 = sp.Eq(d2x_dt2, -dy_dt)
eq2 = sp.Eq(d2y_dt2, dx_dt)

# Initial conditions
x0 = 1.0
y0 = 0.0
dx0_dt = 0
dy0_dt = 1

#dsolve eq1 and eq2
sol= sp.dsolve([eq1, eq2], ics={x(0): x0, y(0): y0, x(t).diff(t).subs(t, 0): dx0_dt, y(t).diff(t).subs(t, 0): dy0_dt})
xSol, ySol = sol[0].rhs, sol[1].rhs

#plot
t_values = np.linspace(0, 10, 100)
x_values = [xSol.subs(t, i) for i in t_values]
y_values = [ySol.subs(t, i) for i in t_values]

plt.plot(t_values, x_values, label='x(t)')
plt.plot(t_values, y_values, label='y(t)')
#%%

# Convert to a system of first-order differential equations
system = [dx_dt, dy_dt, eq1.rhs, eq2.rhs]

# Function to convert the symbolic system to numerical functions
func = sp.lambdify((x(t), y(t), dx_dt, dy_dt), system)

def f_fromSympy(Y,t):
    """Return the right-hand side of the differential equation
    as a numpy array
    """
    return func(Y[0], Y[1], Y[2], Y[3])



# Time array for integration
t_values = np.linspace(0, 10, 100)

# Solve the system using odeint
initial_conditions = [x0, y0, dx0_dt, dy0_dt]
sol = odeint(f_fromSympy, initial_conditions, t_values)

# Extract the x and y values from the solution
x_values, y_values = sol[:, 0], sol[:, 1]

# Plot the results
plt.plot(t_values, x_values, label='x(t)')
plt.plot(t_values, y_values, label='y(t)')
plt.xlabel('t')
plt.ylabel('Values')
plt.legend()
plt.grid()
plt.show()




#%% POSSIBLE SOLUTION WITH 1 DOF
###################################



import numpy as np
import matplotlib.pyplot as plt
h = 1e-6
# Define the Lagrangian function for a simple pendulum

def kinetic(q, q_dot, t, m = 1):
    return 0.5*m*(q_dot**2)

def potential(q, q_dot, t, m = 1, g = 9.81):
        return m*g*(1 - np.cos(q))
        # return m*g*q

def lagrangian(q, q_dot, t):
    m = 1  # Mass of the pendulum
    g = 9.81  # Acceleration due to gravity
    return (kinetic(q, q_dot, t) - potential(q, q_dot, t))*np.exp(0*t)

def firstOrderDerivative(func, q, q_dot, t, h_q = 0, h_qdot = 0, h_t = 0):
    return (func(q + h_q, q_dot + h_qdot, t + h_t) - func(q - h_q, q_dot - h_qdot, t - h_t)) / (2 * ( h_q + h_qdot + h_t))

def secondOrderDerivative(func, q, q_dot, t, h_q = 0, h_qdot = 0, h_t = 0):
    return (func(q + h_q, q_dot + h_qdot, t + h_t) - 2 * func(q, q_dot, t) + func(q - h_q, q_dot - h_qdot, t - h_t)) / (( h_q + h_qdot + h_t)**2)

####
def dL_dq(q, q_dot, t):
    return firstOrderDerivative(lagrangian, q, q_dot, t, h_q = h)

def dL_dqdot(q, q_dot, t):
    return firstOrderDerivative(lagrangian, q, q_dot, t, h_qdot = h)

###


def ODE(y, t):
    q, q_dot = y

    dL_dqdot_dq = firstOrderDerivative(dL_dqdot, q, q_dot, t, h_q = h)
    dL_dqdot_dqdot = secondOrderDerivative(lagrangian, q, q_dot, t, h_qdot = h)
    dL_dqdot_dt = firstOrderDerivative(dL_dqdot, q, q_dot, t, h_t = h)
    q_dot_dot = (dL_dq(q, q_dot, t) -  q_dot*dL_dqdot_dq - dL_dqdot_dt)/(dL_dqdot_dqdot)

    return np.array([q_dot, q_dot_dot])

# Initial conditions
q0 = np.array([np.pi/2*0 + 0.1, 0])



#solve with odeINT
t = np.linspace(0, 3, 1000000)
sol = odeint(ODE, q0, t)
qt = sol[:, 0]
q_dott = sol[:, 1]

#plot the sol
plt.figure()
plt.plot(t, sol[:, 0], label='x')
# plt.plot(t, sol[:, 1], label='Angular Velocity')

analytical = q0[0] * np.cos(np.sqrt(9.81) * t)
plt.plot(t, analytical, label='Analytical')

#total energy
totalEnergy = np.zeros_like(t)
for i in range(len(t)):
    totalEnergy[i] = kinetic(qt[i], q_dott[i], t[i]) + potential(qt[i], q_dott[i], t[i])

plt.plot(t, totalEnergy, label='Total Energy')
# %% IMPLEMENTING WITH MORE THAN 1 DOF
######################################


g = 9.81
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0


import numpy as np
import matplotlib.pyplot as plt
h = 1e-6

# Kinetic energy function
def kinetic(q, v, t):
    q1, q2 = q
    v1, v2 = v
    T1 = 0.5 * m1 * l1**2 * v1**2
    T2 = 0.5 * m2 * (l1**2 * v1**2 + l2**2 * v2**2 + 2 * l1 * l2 * v1 * v2 * np.cos(q1 - q2))
    return T1 + T2

# Potential energy function
def potential(q, v, t):
    q1, q2 = q
    U1 = -m1 * g * l1 * np.cos(q1)
    U2 = -m2 * g * (l1 * np.cos(q1) + l2 * np.cos(q2))
    return U1 + U2


def lagrangian(q, q_dot, t, j = 0):
    return kinetic(q, q_dot, t) - potential(q, q_dot, t)



def firstOrderDerivative(func, q, q_dot, t, i = 0, j = 0, H_q = 0, H_qdot = 0, H_t = 0):
    h_q = np.zeros(2)
    h_qdot = np.zeros(2)

    h_q[i] = H_q
    h_qdot[i] = H_qdot

    return (func(q + h_q, q_dot + h_qdot, t + H_t, j = j) - func(q - h_q, q_dot - h_qdot, t - H_t, j = j)) / (2 * ( H_q + H_qdot + H_t))

def secondOrderDerivative(func, q, q_dot, t, i = 0, j = 0, H_q = 0, H_qdot = 0, H_t = 0):
    h_q = np.zeros(2)
    h_qdot = np.zeros(2)

    h_q[i] = H_q
    h_qdot[i] = H_qdot

    return (func(q + h_q, q_dot + h_qdot, t + H_t, j = j) - 2 * func(q, q_dot, t, j = j) + func(q - h_q, q_dot - h_qdot, t - H_t, j = j)) / (( H_q + H_qdot + H_t)**2)


####
def dL_dq(q, q_dot, t, j = 0):
    test = firstOrderDerivative(lagrangian, q, q_dot, t, i = j, H_q = h)
    return test

def dL_dqdot(q, q_dot, t, j = 0):
    test = firstOrderDerivative(lagrangian, q, q_dot, t, i = j, H_qdot = h)
    return test

###


def ODE(t, y):

    q1, q2, q1_dot, q2_dot = y

    q = np.array([q1, q2])
    q_dot = np.array([q1_dot, q2_dot])

    K1 = dL_dq(q, q_dot, t, j = 0) - firstOrderDerivative(dL_dqdot, q, q_dot, t, j = 0, H_t = h)
    K2 = dL_dq(q, q_dot, t, j = 1) - firstOrderDerivative(dL_dqdot, q, q_dot, t, j = 1, H_t = h)

    A1 = firstOrderDerivative(dL_dqdot, q, q_dot, t, i = 0, j = 0, H_q = h)
    B1 = firstOrderDerivative(dL_dqdot, q, q_dot, t, i = 0, j = 0, H_qdot = h)
    C1 = firstOrderDerivative(dL_dqdot, q, q_dot, t, i = 1, j = 0, H_q = h)
    D1 = firstOrderDerivative(dL_dqdot, q, q_dot, t, i = 1, j = 0, H_qdot = h)

    A2 = firstOrderDerivative(dL_dqdot, q, q_dot, t, i = 0, j = 1, H_q = h)
    B2 = firstOrderDerivative(dL_dqdot, q, q_dot, t, i = 0, j = 1, H_qdot = h)
    C2 = firstOrderDerivative(dL_dqdot, q, q_dot, t, i = 1, j = 1, H_q = h)
    D2 = firstOrderDerivative(dL_dqdot, q, q_dot, t, i = 1, j = 1, H_qdot = h)

    q1_dot_dot = ((A2*q1_dot + C2*q2_dot - K2)*D1 - D2*(A1*q1_dot + C1*q2_dot - K1))/(B1*D2 - B2*D1)
    q2_dot_dot = ((-A2*q1_dot - C2*q2_dot + K2)*B1 + B2*(A1*q1_dot + C1*q2_dot - K1))/(B1*D2 - B2*D1)

    # q1_dot_dot = (K1 - q1_dot*A1 - q2_dot*C1 - ((K2 - q1_dot*A2-q2_dot*C2)/(D2))*D1)/(B1 - (D1*B2/D2))
    # q2_dot_dot = (K2 - q1_dot*A2 - q2_dot*C2 - ((K1 - q1_dot*A1-q2_dot*C1)/(B1))*B2)/(D2 - (B2*D1/B1))
    

    return np.array([q1_dot, q2_dot, q1_dot_dot, q2_dot_dot])

# Initial conditions
q0 = np.array([1,1])
v0 = np.array([3,1])
y0 = np.concatenate((q0, v0))

from scipy.integrate import solve_ivp

tf = 20
t = np.linspace(0, tf, 10000)
#use solve_ivp
sol = solve_ivp(ODE, [0, tf], y0, t_eval=t, method='RK45', rtol = 1e-6, atol = 1e-6)
q1t = sol.y[0]
q2t = sol.y[1]
q1_dott = sol.y[2]
q2_dott = sol.y[3]
t = sol.t



# #solve with odeINT
# t = np.linspace(0, 10, 1000)
# sol = odeint(ODE, y0, t, rtol=1e-12, atol=1e-12)

# q1t = sol[:, 0]
# q2t = sol[:, 1]
# q1_dott = sol[:, 2]
# q2_dott = sol[:, 3]

#plot the sol
plt.figure()
plt.plot(t, q1t, label='Angle 1')
plt.plot(t, q2t, label='Angle 2')
plt.legend()


#%%
#total energy
totalEnergy = np.zeros_like(t)
for i in range(len(t)):
    totalEnergy[i] = kinetic((q1t[i], q2t[i]), (q1_dott[i], q2_dott[i]), t[i]) + potential((q1t[i], q2t[i]), (q1_dott[i], q2_dott[i]), t[i])

initEnergy = totalEnergy[0]
dE = np.abs(totalEnergy - initEnergy)/initEnergy*100

plt.plot(t, dE, label='Total Energy')
