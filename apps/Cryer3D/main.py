import sys
sys.path.insert(0, '../..')

from dolfin import split, plot, interpolate, Expression, Constant, assign
import matplotlib.pyplot as plt

from libs.grids import *
from libs.spaces import *
from libs.io import *
from libs.properties import *
from libs.bcs import *
from libs.linearsystem import *
from libs.solvers import *


""" INPUT """

# Dimensions
x0 = 0.0
y0 = 0.0
z0 = 0.0
R = 1.0
# Refinement
N = 20
# Elements degree
pu = 1
pp = 1
# Simulation time
dt = 1e4
T = 2.5e6
# Load
loadMagnitude = -10.0e3 # in Pa
load = Expression(("load*x[0]/r", "load*x[1]/r", "load*x[2]/r"), r=R, load=loadMagnitude, degree=pu)
# Porous medium
medium = "GulfMexicoShale"
# Input origin
propertiesFolder = "../../data"
propertiesFile = "poroelastic_properties.json"
# Output destination
resultsFolder = "results/P1P1"
resultsFile = "results"
settingsFile = "settings"

""" START """

# Generate grid
grid = QuarterSphereGrid(x0, y0, z0, R, N)
# Generate mixed spaces and trial and test functions
space = DisplacementPressureSpace(grid, pu, pp)
(u, p) = space.trialFunction()
(w, q) = space.testFunction()
# Assign IC
u0 = interpolate(Constant((0.0, 0.0, 0.0)), space.V)
p0 = interpolate(Constant(0.0), space.Q)
# Import porous medium data
properties = getJsonData("{}/{}".format(propertiesFolder, propertiesFile))
properties = PoroelasticProperties(properties[medium])

""" UNDRAINED STEADY-STATE SOLUTION """

# Assign BC
bcs = []
bcs = BoundaryConditions(grid, space)
bcs.addDisplacementHomogeneousBC(2, 0)
bcs.addDisplacementHomogeneousBC(3, 1)
bcs.addDisplacementHomogeneousBC(4, 2)
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
t = 0
writer = XDMFWriter(resultsFolder, resultsFile)
print("Time = {:.3E}".format(t), end="\r")
# Generate independent terms vector
forceVector = ls.forceVector(load, w, 1)
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
u_h.rename("u", "u")
p_h.rename("p", "p")
writer.writeMultiple([u_h, p_h], time=t)
# Next time-step
t += dt
u0.assign(u_h)
p0.assign(p_h)

""" DRAINED TRANSIENT SOLUTION """

# Assign BC
bcs = []
bcs = BoundaryConditions(grid, space)
bcs.addPressureHomogeneousBC(1)
bcs.addDisplacementHomogeneousBC(2, 0)
bcs.addDisplacementHomogeneousBC(3, 1)
bcs.addDisplacementHomogeneousBC(4, 2)
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
while t <= T:
	print("Time = {:.3E}".format(t), end="\r")
	# Generate independent terms vector
	forceVector = ls.forceVector(load, w, 1)
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
	u_h.rename("u", "u")
	p_h.rename("p", "p")
	writer.writeMultiple([u_h, p_h], time=t)
	# Next time-step
	t += dt
	u0.assign(u_h)
	p0.assign(p_h)
writer.close()
# Save simulation data
data = {"Parameters": {"Load": {"Value": loadMagnitude, "Unit": "Pa"}, "Radius": {"Value": R, "Unit": "m"}}, "Simulation": {"Timestep Size": {"Value": dt, "Unit": "s"}, "Total Simulation Time": {"Value": T, "Unit": "s"}, "Refinement": {"Characteristic Length": {"Value": R/N, "Unit": "m"}, "Displacement Elements Degree": pu, "Pressure Elements Degree": pp}}}
saveJsonData(data, resultsFolder, settingsFile)
copyProperties(propertiesFolder, propertiesFile, resultsFolder, [medium])