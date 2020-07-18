from dolfin import Constant
from multiphenics import DirichletBC, BlockDirichletBC


class BoundaryConditions(object):
	def __init__(self, grid, space):
		self.grid = grid
		self.space = space
		self.dirichlet = []

	def addPressureBC(self, mark, value):
		self.dirichlet.append(DirichletBC(self.space.mixedSpace.sub(1), value, self.grid.boundaries, mark))

	def addDisplacementBC(self, mark, value, direction):
		self.dirichlet.append(DirichletBC(self.space.mixedSpace.sub(0).sub(direction), value, self.grid.boundaries, mark))

	def addPressureHomogeneousBC(self, mark):
		self.dirichlet.append(DirichletBC(self.space.mixedSpace.sub(1), Constant(0.), self.grid.boundaries, mark))

	def addDisplacementHomogeneousBC(self, mark, direction):
		self.dirichlet.append(DirichletBC(self.space.mixedSpace.sub(0).sub(direction), Constant(0.), self.grid.boundaries, mark))

	def blockInitialize(self):
		self.dirichlet = BlockDirichletBC([self.dirichlet])