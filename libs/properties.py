class PoroelasticProperties(object):
	def __init__(self, properties):
		self.G = properties["ShearModulus"]
		self.K = properties["BulkModulus"]
		self.Ks = properties["SolidBulkModulus"]
		self.Kf = properties["FluidBulkModulus"]
		self.phi = properties["Porosity"]
		self.k = properties["Permeability"]
		self.mu = properties["FluidViscosity"]
		self.lamda = self.K - 2.*self.G/3.
		self.alpha = 1 - self.K/self.Ks
		self.Q = 1./(self.phi/self.Kf - (self.alpha - self.phi)/self.Ks)
		self.M = self.K + 4.*self.G/3.