from dolfin import Constant
from multiphenics import DirichletBC, BlockDirichletBC


class BoundaryConditions(object):
	def __init__(self, grid, space, split=False):
		self.grid = grid
		self.space = space
		self.split = split
		if self.split:
			self.uDirichlet = []
			self.pDirichlet = []
		else:
			self.dirichlet = []

	def addDisplacementHomogeneousBC(self, mark, direction):
		if self.split:
			self.uDirichlet.append(DirichletBC(self.space.W.sub(direction), Constant(0.), self.grid.boundaries, mark))
		else:
			self.dirichlet.append(DirichletBC(self.space.mixedSpace.sub(0).sub(direction), Constant(0.), self.grid.boundaries, mark))

	def addDisplacementBC(self, mark, value, direction):
		if self.split:
			self.uDirichlet.append(DirichletBC(self.space.W.sub(direction), value, self.grid.boundaries, mark))
		else:
			self.dirichlet.append(DirichletBC(self.space.mixedSpace.sub(0).sub(direction), value, self.grid.boundaries, mark))

	def addPressureHomogeneousBC(self, mark):
		if self.split:
			self.pDirichlet.append(DirichletBC(self.space.Q, Constant(0.), self.grid.boundaries, mark))
		else:
			self.dirichlet.append(DirichletBC(self.space.mixedSpace.sub(1), Constant(0.), self.grid.boundaries, mark))

	def addPressureBC(self, mark, value):
		if self.split:
			self.pDirichlet.append(DirichletBC(self.space.Q, value, self.grid.boundaries, mark))
		else:
			self.dirichlet.append(DirichletBC(self.space.mixedSpace.sub(1), value, self.grid.boundaries, mark))

	def blockInitialize(self):
		if not self.split:
			if self.dirichlet:
				self.dirichlet = BlockDirichletBC([self.dirichlet])