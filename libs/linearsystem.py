from dolfin import inner, grad, sym, div
from multiphenics import block_assemble, block_solve


class LinearSystem(object):
	def __init__(self, grid):
		self.dx = grid.dx
		self.ds = grid.ds
		self.dS = grid.dS
	
	def stiffnessBlock(self, properties, u, w):
		return 2*properties.G*inner(sym(grad(u)), grad(w))*self.dx + properties.lamda*div(u)*div(w)*self.dx

	def porePressureBlock(self, properties, p, w):
		return -properties.alpha*p*div(w)*self.dx

	def solidVelocityBlock(self, properties, dt, u, q):
		return (properties.alpha/dt)*div(u)*q*self.dx

	def storageBlock(self, properties, dt, p, q):
		return (1./(properties.Q*dt))*p*q*self.dx

	def fluidFlowBlock(self, properties, p, q):
		return (properties.k/properties.mu)*inner(grad(p), grad(q))*self.dx

	def forceVector(self, load, w, mark):
		return inner(load, w)*self.ds(mark)

	def apply(self, entity, bcs):
		bcs.apply(entity)
		return entity

	def assembly(self, entity, bcs=[]):
		entity = block_assemble(entity)
		if bcs:
			self.apply(entity, bcs)
		return entity

	def assemblyFullyImplicitMatrix(self, properties, dt, u, w, p, q, bcs=[]):
		stiffnessBlock = self.stiffnessBlock(properties, u, w)
		porePressureBlock = self.porePressureBlock(properties, p, w)
		solidVelocityBlock = self.solidVelocityBlock(properties, dt, u, q)
		storageBlock = self.storageBlock(properties, dt, p, q)
		fluidFlowBlock = self.fluidFlowBlock(properties, p, q)
		A = [[stiffnessBlock, 		porePressureBlock],
			 [solidVelocityBlock,	storageBlock + fluidFlowBlock]]
		if bcs:
			self.fullyImplicitMatrix = self.assembly(A, bcs)
		else:
			self.fullyImplicitMatrix = self.assembly(A)

	def assemblyFullyImplicitVector(self, forceVector):
		f = [forceVector,
		 	 0]
		self.fixedFullyImplicitVector = self.assembly(f)

	def updateFullyImplicitVector(self, properties, dt, u, w, p, q, u0, p0, bcs):
		solidVelocityBlock = self.solidVelocityBlock(properties, dt, u, q)
		storageBlock = self.storageBlock(properties, dt, p, q)
		m = [0,
			 storageBlock*p0]
		m = self.assembly(m)
		s = [0,
			 solidVelocityBlock*u0]
		s = self.assembly(s)
		self.fullyImplicitVector = self.fixedFullyImplicitVector + m + s
		self.fullyImplicitVector = self.apply(self.fullyImplicitVector, bcs)

	def solveFullyImplicitProblem(self, X):
		block_solve(self.fullyImplicitMatrix, X.block_vector(), self.fullyImplicitVector, "mumps")
		self.solution = X