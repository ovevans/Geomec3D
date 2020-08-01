import sys
sys.path.insert(0, '../..')

from dolfin import Expression, Constant, assign

from libs.grids import *
from libs.spaces import *
from libs.io import *
from libs.properties import *
from libs.bcs import *
from libs.linearsystem import *


""" INPUT """

# Dimensions
x0 = -0.5
x1 = 0.5
y0 = -0.5
y1 = 0.5
z0 = 0.0
z1 = 6.0
# Refinement
N = 1
# Elements degree
pu = 1
pp = 1
# Simulation time
dt = 1e5
T = 5e6
# Load
loadMagnitude = -10.0e3 # in Pa
load = Expression(("0.0", "0.0", "load"), load=loadMagnitude, degree=pu)
# Porous medium
medium = "GulfMexicoShale"
# Input origin
propertiesFolder = "../../data"
propertiesFile = "poroelastic_properties.json"
# Output destination
resultsFolder = "results/P1P1"
resultsFile = "results"
settingsFile = "settings"
# Fixed Stress Splitting Scheme
split = False

""" START """

# Generate grid
grid = BoxGrid(x0, x1, y0, y1, z0, z1, N)
# Generate mixed space and trial and test functions
space = CGvCGqSpace(grid, pu, pp, split=split)
(u, p) = space.trialFunction()
(w, q) = space.testFunction()
# Assign IC
(u0, p0) = space.assignInitialCondition(Constant((0.0, 0.0, 0.0)), Constant(0.0))
# Import porous medium data
properties = getJsonData("{}/{}".format(propertiesFolder, propertiesFile))
properties = PoroelasticProperties(properties[medium])

""" UNDRAINED STEADY-STATE SOLUTION """

# Assign BC
bcs = BoundaryConditions(grid, space, split=split)
bcs.addDisplacementHomogeneousBC(1, 0)
bcs.addDisplacementHomogeneousBC(2, 0)
bcs.addDisplacementHomogeneousBC(3, 1)
bcs.addDisplacementHomogeneousBC(4, 1)
bcs.addDisplacementHomogeneousBC(5, 2)
bcs.blockInitialize()
# Generate linear system and coefficients matrix
ls = LinearSystem(grid, split=split)
ls.initializeLinearSystem(properties, dt, u, w, p, q, bcs)
ls.assemblyCoefficientsMatrix()
t = 0
writer = XDMFWriter(resultsFolder, resultsFile)
print("Time = {:.3E}".format(t), end="\r")
# Generate independent terms vector
forceVector = ls.forceVector(load, w, 6)
ls.assemblyVector(forceVector, u0, p0)
# Solve linear system
ls.solveProblem(space.function())
(u_h, p_h) = ls.solution
u_h.rename("u", "u")
p_h.rename("p", "p")
writer.writeMultiple([u_h, p_h], time=t)
# Next time-step
t += dt
u0.assign(u_h)
p0.assign(p_h)

# """ DRAINED TRANSIENT SOLUTION """

# # Assign BC
# bcs = BoundaryConditions(grid, space, split=split)
# bcs.addPressureHomogeneousBC(6)
# bcs.addDisplacementHomogeneousBC(1, 0)
# bcs.addDisplacementHomogeneousBC(2, 0)
# bcs.addDisplacementHomogeneousBC(3, 1)
# bcs.addDisplacementHomogeneousBC(4, 1)
# bcs.addDisplacementHomogeneousBC(5, 2)
# bcs.blockInitialize()
# # Generate linear system and coefficients matrix
# ls = LinearSystem(grid, split=split)
# ls.initializeLinearSystem(properties, dt, u, w, p, q, bcs)
# ls.assemblyCoefficientsMatrix()
# # Generate independent terms vector
# forceVector = ls.forceVector(load, w, 6)
# ls.assemblyVector(forceVector)
# # Loop for transient solution
# while t <= T:
# 	print("Time = {:.3E}".format(t), end="\r")
# 	ls.updateVector(u0, p0)
# 	# Solve linear system
# 	ls.solveProblem(space.function())
# 	(u_h, p_h) = ls.getSolution()
# 	u_h.rename("u", "u")
# 	p_h.rename("p", "p")
# 	writer.writeMultiple([u_h, p_h], time=t)
# 	# Next time-step
# 	t += dt
# 	u0.assign(u_h)
# 	p0.assign(p_h)
# writer.close()
# # Save simulation data
# data = {"Parameters": {"Load": {"Value": loadMagnitude, "Unit": "Pa"}, "Dimensions": {"Length": {"Axis": "x", "Value": x1 - x0, "Unit": "m"}, "Width": {"Axis": "y", "Value": y1 - y0, "Unit": "m"}, "Height": {"Axis": "z", "Value": z1 - z0, "Unit": "m"}}}, "Simulation": {"Timestep Size": {"Value": dt, "Unit": "s"}, "Total Simulation Time": {"Value": T, "Unit": "s"}, "Refinement": {"Resolution": N, "Displacement Elements Degree": pu, "Pressure Elements Degree": pp}}}
# saveJsonData(data, resultsFolder, settingsFile)
# copyProperties(propertiesFolder, propertiesFile, resultsFolder, [medium])