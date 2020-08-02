from dolfin import inner, grad, sym, div, assemble, Constant, errornorm, norm, set_log_level, PETScKrylovSolver
from multiphenics import block_assemble, block_solve


class LinearSystem(object):
	def __init__(self, grid, split=False, tol=1.e-6, maxIteration=100):
		self.dx = grid.dx
		self.ds = grid.ds
		self.dS = grid.dS
		self.tol = tol
		self.maxIteration = maxIteration
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

	def forceVector(self, properties, load, w, mark, g=Constant((0., 0., 0.))):
		return inner(load, w)*self.ds(mark) + properties.rho*inner(g, w)*self.dx

	def hydrostatVector(self, properties, g, q):
		return (properties.k/properties.mu)*properties.rho_f*inner(g, grad(q))*self.dx

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
		if self.split:
			self.geoSolver = PETScKrylovSolver("gmres", "hypre_amg")
			self.geoSolver.parameters['relative_tolerance'] = self.tol
			self.flowSolver = PETScKrylovSolver("bicgstab", "sor")
			self.flowSolver.parameters['relative_tolerance'] = self.tol
		"""
		Solvers:
		'bicgstab'	Biconjugate gradient stabilized method
		'cg'	Conjugate gradient method
		'gmres'	Generalized minimal residual method
		'minres'	Minimal residual method
		'petsc'	PETSc built in LU solver
		'richardson'	Richardson method
		'superlu_dist'	Parallel SuperLU
		'tfqmr'	Transpose-free quasi-minimal residual method
		'umfpack'	UMFPACK
		Preconditioners:
		'icc'	Incomplete Cholesky factorization
		'ilu'	Incomplete LU factorization
		'petsc_amg'	PETSc algebraic multigrid
		'hypre_amg'	HYPRE algebraic multigrid
		'sor'	Successive over-relaxation
		"""

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

	def assemblyVector(self, forceVector, u0, p0, hydrostatVector=False):
		if self.split:
			self.u0 = u0
			self.p0 = p0
			self.fixedGeoVector = self.assembly(forceVector)
			if hydrostatVector:
				self.fixedflowVector = self.assembly(self.storageBlock*p0) + self.assembly(self.solidVelocityBlock*u0) + self.assembly(hydrostatVector)
			else:
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
			if hydrostatVector:
				g = [0,
					 hydrostatVector]
				g = self.assembly(g)
				self.vector = self.apply(f + m + s + g, self.bcs.dirichlet)
			else:
				self.vector = self.apply(f + m + s, self.bcs.dirichlet)

	def iterateGeoVector(self, p_hk):
		self.geoVector = self.fixedGeoVector + self.assembly(-self.porePressureBlock*p_hk)

	def iterateFlowVector(self, u_hk, p_hk):
		self.flowVector = self.fixedflowVector + self.assembly(-self.solidVelocityBlock*u_hk) + self.assembly(self.solidPressureBlock*p_hk)

	def getRelativeErrorNorm(self, p_h, p_hk):
		set_log_level(40)
		error = errornorm(p_h, p_hk)/norm(p_h)
		set_log_level(20)
		return error

	def solveProblem(self, space):
		X = space.function()
		if self.split:
			(u_h, p_h) = X
			(u_hk, p_hk) = space.assignInitialCondition(Constant((0.0, 0.0, 0.0)), Constant(0.0))
			u_hk.assign(self.u0)
			p_hk.assign(self.p0)
			error = 1e34
			iteration = 0
			while error > self.tol and iteration < self.maxIteration:
				print("Iteration = {:4d}".format(iteration + 1), end="\t")
				self.iterateFlowVector(u_hk, p_hk)
				self.apply(self.flowVector, self.bcs.pDirichlet)
				self.flowSolver.solve(self.flowCoefficientsMatrix, p_h.vector(), self.flowVector)
				p_hk.assign(p_h)
				self.iterateGeoVector(p_hk)
				self.apply(self.geoVector, self.bcs.uDirichlet)
				self.geoSolver.solve(self.geoCoefficientsMatrix, u_h.vector(), self.geoVector)
				u_hk.assign(u_h)
				self.iterateFlowVector(u_hk, p_hk)
				self.apply(self.flowVector, self.bcs.pDirichlet)
				self.flowSolver.solve(self.flowCoefficientsMatrix, p_h.vector(), self.flowVector)
				error = self.getRelativeErrorNorm(p_h, p_hk)
				print("Error = {:.2E}".format(error), end="\r")
				p_hk.assign(p_h)
				iteration += 1
			print("\033[K", end="\r")
			print("Iteration = {:4d}".format(iteration + 1), end="\t")
			print("Error = {:.2E}".format(error), end="\t")
			self.solution = (u_h, p_h)
		else:
			block_solve(self.coefficientsMatrix, X.block_vector(), self.vector, "mumps")
			self.solution = X.block_split()