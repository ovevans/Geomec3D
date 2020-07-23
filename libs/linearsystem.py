from dolfin import inner, grad, sym, div
from multiphenics import block_assemble


class LinearSystem(object):
	def __init__(self, grid):
		self.dx = grid.dx
		self.ds = grid.ds
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

	def accumulationVector(self, properties, dt, p0, u0, q):
		# return (properties.alpha/dt)*div(u0)*q*self.dx + (1./(properties.Q*dt))*p0*self.dx
		return 0

	def assembly(self, entity, bcs):
		entity = block_assemble(entity)
		if bcs:
			bcs.apply(entity)
		return entity