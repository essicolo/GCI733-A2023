import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import fipy as fp
PETSc.Options().setValue("snes_type", "newtonls")

def vg_k(u, k_sat, a, n, m, l=0.5):
    k = np.where(
        u>=0,
        k_sat,
        k_sat * (1 - (-a * u)**(n-1) * (1 + (-a * u)**n)**(-m))**2 * (1 + (-a * u)**n)**(-m*l)
    )
    return k

def find_u_for_given_k(target_k, k_sat, a, n, m, l=0.5, u_guess=-1.0):
    def f(u):
        return vg_k(u, k_sat, a, n, m, l) - target_k
    u_value, = fsolve(f, u_guess)
    return u_value

def linear_method(z, u_q):
    u = np.where(z < -u_q, -z, u_q)
    return u

def kisch(q, u, u_q, k_sat, a, n, m, l=0.5, z_base=0):
    du = -np.array([u[i+1] - u[i] for i in range(len(u)-1)] + [np.nan])
    k = vg_k(u, k_sat, a, n, m, l)
    z = (du / (1 - q/k)).cumsum() + z_base
    return z

def richards_fem(q, k_sat, a, n, m, l=0.5, Lz=10, nz=100, max_iters=100, tolerance=1E-8):

    # Domain
    dz = Lz / nz
    mesh = fp.Grid1D(nx=nz, dx=dz)

    # Variables
    z = fp.CellVariable(name="elevation", mesh=mesh, value=mesh.cellCenters[0])
    h = fp.CellVariable(name="total head", mesh=mesh, value=0.0, hasOld=True)
    u = fp.CellVariable(name="matrix pressure", mesh=mesh, value=h.value-z.value, hasOld=True)    
    k = fp.CellVariable(mesh=mesh, value=vg_k(u.value, k_sat, a, n, m, l))

    # Boundary conditions
    h.constrain(0.0, mesh.facesLeft) # Dirichlet at bottom
    h.faceGrad.constrain(- q/k.faceValue, where=mesh.facesRight)  # Neumann at top

    # Equation
    eq = (fp.DiffusionTerm(coeff=k, var=h) == 0)

    # Solve with Picard iterations
    solver = fp.DefaultSolver(tolerance=1e-10, iterations=5000)
    convergence = []
    for iter in range(max_iters):
        u.value = h.value - z.value
        k.value = vg_k(u.value, k_sat, a, n, m, l)
        eq.sweep(var=h, solver=solver)
        error = np.max(np.abs(h.value - h.old.value))
        convergence.append(error)
        if error < tolerance:
            break

    # update u for final h
    u.value = h.value - z.value

    return u.value, h.value, z.value, convergence

a = 1.0  # [1/m]
n = 2.5
m = 1 - 1/n
l = 0.5
k_sat = 1.0E-4  # [m/s]
rain = 1E-6

u = -np.logspace(-3, 2, 100)
k = vg_k(u, k_sat=k_sat, a=a, n=n, m=m, l=l)
u_q = find_u_for_given_k(rain, k_sat, a, n, m, l)

z_linear = np.linspace(0, 5, 100)
u_linear = linear_method(z_linear, u_q)

u_kisch = np.linspace(0, u_q, 1000)
z_kisch = kisch(q=rain, u=u_kisch, u_q=u_q, k_sat=k_sat, a=a, n=n, m=m, l=l, z_base=0)

u_fem, h_fem, z_fem, convergence = richards_fem(
    q=rain, k_sat=k_sat, a=a, n=n, m=m, l=l,
    Lz=10, nz=100, max_iters=100, tolerance=1e-12
)

plt.plot(u_linear, z_linear, label='Linear method')
plt.plot(u_kisch, z_kisch, label="Kisch method")
plt.plot(u_fem, z_fem, label="FE, Fipy")
plt.title('Pressure profile')
plt.legend()

print(convergence)