from dolfin import inner, grad, sym, div, DOLFIN_EPS, assemble, solve, sqrt
from multiphenics import block_assemble, block_solve


class LinearSystem(object):
	def __init__(self, grid, tol=DOLFIN_EPS, split=False):
		self.dx = grid.dx
		self.ds = grid.ds
		self.dS = grid.dS
		self.tol = tol
		self.split = split
	
	def stiffnessBlock(self, properties, u, w):
		return 2*properties.G*inner(sym(grad(u)), grad(w))*self.dx + properties.lamda*div(u)*div(w)*self.dx

	def porePressureBlock(self, properties, p, w):
		return -properties.alpha*p*div(w)*self.dx

	def solidVelocityBlock(self, properties, dt, u, q):
		return (properties.alpha/dt)*div(u)*q*self.dx

	def storageBlock(self, properties, dt, p, q):
		return (1./(properties.Q*dt))*p*q*self.dx

	def solidPressureBlock(self, properties, dt, p, q, delta=1):
		return (properties.alpha**2/(delta*properties.K*dt))*p*q*self.dx

	def fluidFlowBlock(self, properties, p, q):
		return (properties.k/properties.mu)*inner(grad(p), grad(q))*self.dx

	def forceVector(self, load, w, mark):
		return inner(load, w)*self.ds(mark)

	def apply(self, entity, bcs):
		if self.split:
			for bc in bcs:
				bc.apply(entity)
		else:
			bcs.apply(entity)
		return entity

	def assembly(self, entity, bcs=[]):
		if self.split:
			entity = assemble(entity)
		else:
			entity = block_assemble(entity)
		if bcs:
			self.apply(entity, bcs)
		return entity

	def initializeLinearSystem(self, properties, dt, u, w, p, q, bcs):
		self.stiffnessBlock = self.stiffnessBlock(properties, u, w)
		self.porePressureBlock = self.porePressureBlock(properties, p, w)
		self.solidVelocityBlock = self.solidVelocityBlock(properties, dt, u, q)
		self.storageBlock = self.storageBlock(properties, dt, p, q)
		self.solidPressureBlock = self.solidPressureBlock(properties, dt, p, q)
		self.fluidFlowBlock = self.fluidFlowBlock(properties, p, q)
		self.bcs = bcs

	def assemblyCoefficientsMatrix(self):
		if self.split:
			K = self.stiffnessBlock
			M = self.storageBlock + self.solidPressureBlock + self.fluidFlowBlock
			self.geoCoefficientsMatrix = self.assembly(K, self.bcs.uDirichlet)
			self.flowCoefficientsMatrix = self.assembly(M, self.bcs.pDirichlet)
		else:
			A = [[self.stiffnessBlock, 		self.porePressureBlock],
				 [self.solidVelocityBlock,	self.storageBlock + self.fluidFlowBlock]]
			self.coefficientsMatrix = self.assembly(A, self.bcs.dirichlet)

	def assemblyVector(self, forceVector, u0, p0):
		if self.split:
			self.u0 = u0
			self.p0 = p0
			self.fixedGeoVector = self.assembly(forceVector)
			self.fixedflowVector = self.assembly(self.storageBlock*p0) + self.assembly(self.solidVelocityBlock*u0)
		else:
			f = [forceVector,
			 	 0]
			f = self.assembly(f)
			m = [0,
				 self.storageBlock*p0]
			m = self.assembly(m)
			s = [0,
				 self.solidVelocityBlock*u0]
			s = self.assembly(s)
			self.vector = self.apply(f + m + s, self.bcs.dirichlet)

	def iterateGeoVector(self, pk):
		self.geoVector = self.fixedGeoVector + self.assembly(-self.porePressureBlock*pk)

	def iterateFlowVector(self, uk, pk):
		self.flowVector = self.fixedflowVector + self.assembly(-self.solidVelocityBlock*uk) + self.assembly(self.solidPressureBlock*pk)

	def solveProblem(self, X):
		if self.split:
			(u, p) = X
			(uk, pk) = X
			p.assign(self.p0)
			error =1e34
			while error > self.tol:
				pk.assign(p)
				self.iterateGeoVector(pk)
				self.apply(self.geoVector, self.bcs.uDirichlet)
				solve(self.geoCoefficientsMatrix, u.vector(), self.geoVector, "mumps")
				uk.assign(u)
				self.iterateFlowVector(uk, pk)
				self.apply(self.flowVector, self.bcs.pDirichlet)
				solve(self.flowCoefficientsMatrix, p.vector(), self.flowVector, "mumps")
				error = sqrt(assemble(inner(grad(pk - p), grad(pk -p))*self.dx))
			self.solution = (u, p)
		else:
			block_solve(self.coefficientsMatrix, X.block_vector(), self.vector, "mumps")
			self.solution = X.block_split()