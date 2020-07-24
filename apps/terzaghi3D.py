import sys
sys.path.insert(0, '../')

from dolfin import split, plot, interpolate, Constant, assign
import matplotlib.pyplot as plt

from libs.grids import *
from libs.spaces import *
from libs.xdmf2dolfin import *
from libs.io import *
from libs.properties import *
from libs.bcs import *
from libs.linearsystem import *
from libs.solvers import *


""" INPUT """

# Dimensions
x0 = 0.0
x1 = 1.0
y0 = 0.0
y1 = 1.0
z0 = 0.0
z1 = 6.0
# Refinement
Nx = 2
Ny = 2
Nz = 12
# Elements degree
pu = 1
pp = 1
# Simulation time
dt = 1e5
T = 2.5e7
# Load
load = Constant((0.0, 0.0, -10.0e3))
# Porous medium
medium = "GulfMexicoShale"

""" START """

# Generate grid
grid = BoxGrid(x0, x1, y0, y1, z0, z1, Nx, Ny, Nz)
# Generate mixed spaces and trial and test functions
space = DisplacementPressureSpace(grid, pu, pp)
(u, p) = space.trialFunction()
(w, q) = space.testFunction()
# Assign IC
u0 = interpolate(Constant((0.0, 0.0, 0.0)), space.V)
p0 = interpolate(Constant(0.0), space.Q)
solution_u = {}
solution_p = {}
solution_u[0] = u0
solution_p[0] = p0
u0.rename("u", "u")
p0.rename("p", "p")
writer = XDMFWriter("results", "terzaghi3D")
writer.writeMultiple([u0, p0])
# Import porous medium data
properties = getJsonData("../data/poroelastic_properties.json")
properties = PoroelasticProperties(properties[medium])
# Assign BC
bcs = []
bcs = BoundaryConditions(grid, space)
bcs.addPressureHomogeneousBC(6)
bcs.addDisplacementHomogeneousBC(1, 0)
bcs.addDisplacementHomogeneousBC(2, 0)
bcs.addDisplacementHomogeneousBC(3, 1)
bcs.addDisplacementHomogeneousBC(4, 1)
bcs.addDisplacementHomogeneousBC(5, 2)
bcs.blockInitialize()
# Generate linear system and coefficients matrix
ls = LinearSystem(grid)
stiffnessBlock = ls.stiffnessBlock(properties, u, w)
porePressureBlock = ls.porePressureBlock(properties, p, w)
solidVelocityBlock = ls.solidVelocityBlock(properties, dt, u, q)
storageBlock = ls.storageBlock(properties, dt, p, q)
fluidFlowBlock = ls.fluidFlowBlock(properties, p, q)
A = [[stiffnessBlock, 		porePressureBlock],
	 [solidVelocityBlock,	storageBlock + fluidFlowBlock]]
A = ls.assembly(A, bcs.dirichlet)
# Loop for transient solution
t = dt
while t <= T:
	print("Time = {:.3E}".format(t), end="\r")
	# Generate independent terms vector
	forceVector = ls.forceVector(load, w, 6)
	f = [forceVector,
	 	 0]
	f = ls.assembly(f)
	m = [0,
		 storageBlock*p0]
	m = ls.assembly(m)
	s = [0,
		 solidVelocityBlock*u0]
	s = ls.assembly(s)
	b = f + m + s
	b = ls.apply(b, bcs.dirichlet)
	# Solve linear system
	solver = Mumps(A, space.function(), b)
	(u_h, p_h) = solver.solution.block_split()
	solution_u[t] = u_h
	solution_p[t] = p_h
	u_h.rename("u", "u")
	p_h.rename("p", "p")
	writer.writeMultiple([u_h, p_h], time=t)
	# Next time-step
	t += dt
	u0.assign(u_h)
	p0.assign(p_h)
writer.close()