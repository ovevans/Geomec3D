from dolfin import VectorElement, FiniteElement, FunctionSpace
from multiphenics import BlockFunctionSpace, BlockTrialFunction, BlockTestFunction, BlockFunction, block_split


class CGvCGqSpace(object):
	def __init__(self, grid, u_degree, p_degree):
		v_elem = VectorElement("CG", grid.mesh.ufl_cell(), u_degree)
		q_elem = FiniteElement("CG", grid.mesh.ufl_cell(), p_degree)
		self.V = FunctionSpace(grid.mesh, v_elem)
		self.Q = FunctionSpace(grid.mesh, q_elem)
		self.mixedSpace = BlockFunctionSpace([self.V, self.Q])

	def trialFunction(self):
		return block_split(BlockTrialFunction(self.mixedSpace))

	def testFunction(self):
		return block_split(BlockTestFunction(self.mixedSpace))

	def function(self):
		return BlockFunction(self.mixedSpace)