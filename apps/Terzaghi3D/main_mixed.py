import sys
sys.path.insert(0, '../..')

from dolfin import split, plot, interpolate, Constant, assign
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
x0 = -0.5
x1 = 0.5
y0 = -0.5
y1 = 0.5
z0 = 0.0
z1 = 6.0
# Refinement
Nx = 4
Ny = 4
Nz = 24
# Elements degree
pu = 1
pp = 1
psig = 1
# Simulation time
dt = 1e5
T = 5e7
# Load
loadMagnitude = -10.0e3 # in Pa
load = Constant((0.0, 0.0, loadMagnitude))
# Porous medium
medium = "GulfMexicoShale"
# Input origin
propertiesFolder = "../../data"
propertiesFile = "poroelastic_properties.json"
# Output destination
resultsFolder = "results/P1P1P1"
resultsFile = "results"
settingsFile = "settings"

""" START """

# Generate grid
grid = BoxGrid(x0, x1, y0, y1, z0, z1, Nx, Ny, Nz)
# Generate mixed spaces and trial and test functions
space = DisplacementPressureVolStressSpace(grid, pu, pp, psig)
(u, p, sig) = space.trialFunction()
(w, q, tau) = space.testFunction()
# Assign IC
u0 = interpolate(Constant((0.0, 0.0, 0.0)), space.V)
p0 = interpolate(Constant(0.0), space.Q)
sig0 = interpolate(Constant(0.0), space.T)
# Import porous medium data
properties = getJsonData("{}/{}".format(propertiesFolder, propertiesFile))
properties = PoroelasticProperties(properties[medium])

""" UNDRAINED STEADY-STATE SOLUTION """

# Assign BC
bcs = []
bcs = BoundaryConditions(grid, space)
bcs.addDisplacementHomogeneousBC(1, 0)
bcs.addDisplacementHomogeneousBC(2, 0)
bcs.addDisplacementHomogeneousBC(3, 1)
bcs.addDisplacementHomogeneousBC(4, 1)
bcs.addDisplacementHomogeneousBC(5, 2)
bcs.blockInitialize()
# Generate linear system and coefficients matrix
ls = LinearSystem(grid)
shearStressBlock = ls.shearStressBlock(properties, u, w)
solidPressureBlock = ls.solidPressureBlock(sig, w)
stressStorageBlock = ls.stressStorageBlock(properties, dt, p, q)
fluidFlowBlock = ls.fluidFlowBlock(properties, p, q)
stressVelocityBlock = ls.stressVelocityBlock(properties, dt, sig, q)
volStrainBlock = ls.volStrainBlock(properties, u, tau)
volPressureBlock = ls.volPressureBlock(properties, p, tau)
volStressBlock = ls.volStressBlock(sig, tau)
A = [[shearStressBlock, 	0,										solidPressureBlock],
	 [0,					stressStorageBlock + fluidFlowBlock,	stressVelocityBlock],
	 [volStrainBlock,		volPressureBlock,						volStressBlock]]
A = ls.assembly(A, bcs.dirichlet)
t = 0
writer = XDMFWriter(resultsFolder, resultsFile)
print("Time = {:.3E}".format(t), end="\r")
# Generate independent terms vector
forceVector = ls.forceVector(load, w, 6)
f = [forceVector,
 	 0,
 	 0]
f = ls.assembly(f)
m = [0,
	 stressStorageBlock*p0,
	 0]
m = ls.assembly(m)
s = [0,
	 stressVelocityBlock*sig0,
	 0]
s = ls.assembly(s)
b = f + m + s
b = ls.apply(b, bcs.dirichlet)
# Solve linear system
solver = Mumps(A, space.function(), b)
(u_h, p_h, sig_h) = solver.solution.block_split()
u_h.rename("u", "u")
p_h.rename("p", "p")
sig_h.rename("sig", "sig")
writer.writeMultiple([u_h, p_h, sig_h], time=t)
# Next time-step
t += dt
u0.assign(u_h)
p0.assign(p_h)
sig0.assign(sig_h)

""" DRAINED TRANSIENT SOLUTION """

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
shearStressBlock = ls.shearStressBlock(properties, u, w)
solidPressureBlock = ls.solidPressureBlock(sig, w)
stressStorageBlock = ls.stressStorageBlock(properties, dt, p, q)
fluidFlowBlock = ls.fluidFlowBlock(properties, p, q)
stressVelocityBlock = ls.stressVelocityBlock(properties, dt, sig, q)
volStrainBlock = ls.volStrainBlock(properties, u, tau)
volPressureBlock = ls.volPressureBlock(properties, p, tau)
volStressBlock = ls.volStressBlock(sig, tau)
A = [[shearStressBlock, 	0,										solidPressureBlock],
	 [0,					stressStorageBlock + fluidFlowBlock,	stressVelocityBlock],
	 [volStrainBlock,		volPressureBlock,						volStressBlock]]
A = ls.assembly(A, bcs.dirichlet)
# Loop for transient solution
while t <= T:
	print("Time = {:.3E}".format(t), end="\r")
	# Generate independent terms vector
	forceVector = ls.forceVector(load, w, 6)
	f = [forceVector,
	 	 0,
	 	 0]
	f = ls.assembly(f)
	m = [0,
		 stressStorageBlock*p0,
		 0]
	m = ls.assembly(m)
	s = [0,
		 stressVelocityBlock*sig0,
		 0]
	s = ls.assembly(s)
	b = f + m + s
	b = ls.apply(b, bcs.dirichlet)
	# Solve linear system
	solver = Mumps(A, space.function(), b)
	(u_h, p_h, sig_h) = solver.solution.block_split()
	u_h.rename("u", "u")
	p_h.rename("p", "p")
	sig_h.rename("sig", "sig")
	writer.writeMultiple([u_h, p_h, sig_h], time=t)
	# Next time-step
	t += dt
	u0.assign(u_h)
	p0.assign(p_h)
	sig0.assign(sig_h)
writer.close()
# Save simulation data
data = {"Parameters": {"Load": {"Value": loadMagnitude, "Unit": "Pa"}, "Dimensions": {"Length": {"Axis": "x", "Value": x1 - x0, "Unit": "m"}, "Width": {"Axis": "y", "Value": y1 - y0, "Unit": "m"}, "Height": {"Axis": "z", "Value": z1 - z0, "Unit": "m"}}}, "Simulation": {"Timestep Size": {"Value": dt, "Unit": "s"}, "Total Simulation Time": {"Value": T, "Unit": "s"}, "Refinement": {"Characteristic Length": {"Value": ((x1 - x0)/Nx*(y1 - y0)/Ny*(z1 - z0)/Nz)**(1/3), "Unit": "m"}, "Displacement Elements Degree": pu, "Pressure Elements Degree": pp, "Volumetric Stress Elements Degree": psig}}}
saveJsonData(data, resultsFolder, settingsFile)
copyProperties(propertiesFolder, propertiesFile, resultsFolder, [medium])