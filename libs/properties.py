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
		self.Q = 1./(self.phi/self.Kf + (self.alpha - self.phi)/self.Ks)
		self.M = self.lamda + 2*self.G
		self.rho_f = 0
		self.rho_s = 0
		self.rho = 0

	def loadDensityData(self, data):
		self.rho_f = data["FluidDensity"]
		self.rho_s = data["SolidDensity"]
		self.rho = self.phi*self.rho_f + (1 - self.phi)*self.rho_s