from dolfin import inner, grad, sym, div
from multiphenics import block_assemble


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

	def assembly(self, entity, bcs = []):
		entity = block_assemble(entity)
		if bcs:
			self.apply(entity, bcs)
		return entity

	def shearStressBlock(self, properties, u, w):
		return 2*properties.G*inner(sym(grad(u)), grad(w))*self.dx

	def solidPressureBlock(self, sig, w):
		return sig*div(w)*self.dx

	def stressStorageBlock(self, properties, dt, p, q):
		return (1./(properties.Q*dt) + properties.alpha**2/(properties.lamda*dt))*p*q*self.dx

	def stressVelocityBlock(self, properties, dt, sig, q):
		return (properties.alpha/(properties.lamda*dt))*sig*q*self.dx

	def volStrainBlock(self, properties, u, tau):
		return -properties.lamda*div(u)*tau*self.dx

	def volPressureBlock(self, properties, p, tau):
		return properties.alpha*p*tau*self.dx

	def volStressBlock(self, sig, tau):
		return sig*tau*self.dx