from dolfin import VectorElement, FiniteElement, FunctionSpace, TrialFunction, TestFunction, Function, interpolate
from multiphenics import BlockFunctionSpace, BlockTrialFunction, BlockTestFunction, BlockFunction, block_split


class CGvCGqSpace(object):
	def __init__(self, grid, u_degree, p_degree, split=False):
		w_elem = VectorElement("CG", grid.mesh.ufl_cell(), u_degree)
		q_elem = FiniteElement("CG", grid.mesh.ufl_cell(), p_degree)
		self.W = FunctionSpace(grid.mesh, w_elem)
		self.Q = FunctionSpace(grid.mesh, q_elem)
		self.split = split
		if not self.split:
			self.mixedSpace = BlockFunctionSpace([self.W, self.Q])

	def trialFunction(self):
		if self.split:
			return (TrialFunction(self.W), TrialFunction(self.Q))
		else:
			return block_split(BlockTrialFunction(self.mixedSpace))

	def testFunction(self):
		if self.split:
			return (TestFunction(self.W), TestFunction(self.Q))
		else:
			return block_split(BlockTestFunction(self.mixedSpace))

	def function(self):
		if self.split:
			return (Function(self.W), Function(self.Q))
		else:
			return BlockFunction(self.mixedSpace)

	def assignInitialCondition(self, displacement, pressure):
		u0 = interpolate(displacement, self.W)
		p0 = interpolate(pressure, self.Q)
		return (u0, p0)