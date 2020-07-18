from dolfin import inner, sym, grad, div, dx, ds
from multiphenics import block_assemble
import numpy as np


class LinearSystem(object):
	def stiffnessBlock(self, properties, u, w):
		return 2*properties.G*inner(sym(grad(u)), grad(w))*dx + properties.lamda*div(u)*div(w)*dx

	def porePressureBlock(self, properties, p, w):
		return -2*properties.alpha*p*div(w)*dx

	def solidVelocityBlock(self, properties, u, q):
		return properties.alpha*div(u)*q*dx

	def storageBlock(self, properties, p, q):
		return (1./properties.Q)*p*q*dx

	def fluidFlowBlock(self, properties, p, q):
		return (properties.k/properties.mu)*inner(grad(p), grad(q))*dx

	def forceVector(self, load, w, mark):
		return inner(load, w)*ds(mark)

	def assembly(self, entity, bcs):
		entity = block_assemble(entity)
		bcs.apply(entity)
		return entity