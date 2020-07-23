import sys
sys.path.insert(0, '../')

from dolfin import split, plot, interpolate, Constant
import matplotlib.pyplot as plt

from libs.grids import *
from libs.spaces import *
from libs.io import *
from libs.properties import *
from libs.bcs import *
from libs.linearsystem import *
from libs.solvers import *
from libs.xdmf2dolfin import *


grid = RectangleGrid(0., 1., 0., 6., 2, 12)
dt = 1e5
T = 1e8

space = DisplacementPressureSpace(grid, 1, 1)
(u, p) = space.trialFunction()
(w, q) = space.testFunction()

u0 = interpolate(Constant((0.0, 0.0)), space.V)
p0 = interpolate(Constant(0.0), space.Q)

properties = getJsonData("../data/poroelastic_properties.json")
properties = PoroelasticProperties(properties["GulfMexicoShale"])

bcs = []
bcs = BoundaryConditions(grid, space)
# bcs.addPressureHomogeneousBC(4)
bcs.addDisplacementHomogeneousBC(1, 0)
bcs.addDisplacementHomogeneousBC(2, 0)
bcs.addDisplacementHomogeneousBC(3, 1)
bcs.blockInitialize()

ls = LinearSystem(grid)
stiffnessBlock = ls.stiffnessBlock(properties, u, w)
porePressureBlock = ls.porePressureBlock(properties, p, w)
solidVelocityBlock = ls.solidVelocityBlock(properties, dt, u, q)
storageBlock = ls.storageBlock(properties, dt, p, q)
fluidFlowBlock = ls.fluidFlowBlock(properties, p, q)
A = [[stiffnessBlock, 		porePressureBlock],
	 [solidVelocityBlock,	storageBlock + fluidFlowBlock]]
A = ls.assembly(A, bcs.dirichlet)

# for i,row in enumerate(A.array()):
# 	print("Row {}: ".format(i + 1), end="")
# 	for j,elem in enumerate(row):
# 		print("({}, {:.2E}) ".format(j, elem), end="")
# 	print("\n")

X = space.function()
t = dt
while t <= T:
	print("Time = {:.3E}".format(t), end="\r")
	forceVector = ls.forceVector(Constant((0., -10.e3)), w, 4)
	accumulationVector = ls.accumulationVector(properties, dt, p0, u0, q)
	b = [forceVector,
	 	 accumulationVector]
	b = ls.assembly(b, bcs.dirichlet)
	solver = Mumps(A, X, b)
	(u_h, p_h) = solver.solution.block_split()
	t += dt
	u0.assign(u_h)
	p0.assign(p_h)
	
u_h.rename("u", "u")
p_h.rename("p", "p")
# plt.figure(1)
# c = plot(p_h, title="Pressure")
# # plt.colorbar(c)
# plt.show()
XDMFFieldWriter("results", "terzaghi2D", [u_h, p_h])