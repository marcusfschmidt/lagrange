# lagrange
Various bits of code related to analytical mechanics.

figSetup defines a plotting module to ensure consistent plots between files.
plotstyle.mplstyle is a custom plotting style for matplotlib.

# Scripts
In /scripts, various small scripts can be found meant to solve analytical mechanics problems.\
lagrange_n=1.py solves an arbitrary Lagrangian with dissipation with a single generalized coordinate, such as a single pendulum or a damped mass-spring system. The script makes no use of SymPy and implements central-difference approximations for the derivatives fo the Lagrangian needed to obtain the equations of motion.