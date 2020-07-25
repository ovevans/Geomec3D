from dolfin import near
import math

from libs.properties import *


class Cryer(object):
	def __init__(self, properties, settings, numOfRoots=100, maxIterations=100, minResidue=1.e-8):
		properties = PoroelasticProperties(properties)
		self.load = settings["Parameters"]["Load"]["Value"]
		self.radius = settings["Parameters"]["Radius"]["Value"]
		self.eta = (properties.K + 4./3.*properties.G)/(2.*properties.G)*(1. + properties.K/(properties.Q*properties.alpha**2))
		self.c = (properties.k*properties.Q*properties.M)/(properties.mu*(properties.M + properties.alpha**2*properties.Q))
		self.p0 = -(properties.alpha*properties.Q*self.load)/(properties.K + properties.alpha**2*properties.Q)
		self.numOfRoots = numOfRoots
		self.maxIterations = maxIterations
		self.minResidue = minResidue
		self.roots = self.computeRoots()

	def computeRoots(self):
		roots = []
		pi = math.pi
		for i in range(self.numOfRoots):
			x1 = (i + 1)*pi - pi/3.
			x2 = (i + 1)*pi + pi/3.
			x3 = (x1 + x2)/2.
			y3 = (1 - self.eta*x3*x3/2.)*math.tan(x3) - x3
			iteration = 0
			residue = 1.
			while iteration < self.maxIterations and residue > self.minResidue and not near(y3, 0.):
				y1 = (1 - self.eta*x1*x1/2.)*math.tan(x1) - x1
				y3 = (1 - self.eta*x3*x3/2.)*math.tan(x3) - x3
				if y1*y3 > 0.0:
					x1 = x3
				else:
					x2 = x3
				x3 = (x1 + x2)/2.
				residue = x2 - x1
				iteration += 1
			roots.append(x3)
		return roots

	def exactSolution(self, time):
		if near(time, 0.):
			return self.p0
		else:
			summation = 0
			for i in range(self.numOfRoots):
				root = self.roots[i]
				num = (math.sin(root) - root)*math.exp(-root**2*self.c*time/self.radius**2)
				den = self.eta*root*math.cos(root)/2. + (self.eta - 1)*math.sin(root)
				summation += self.eta*(num/den)
			return summation*self.p0


class Terzaghi(object):
	def __init__(self, properties, settings, numOfTerms=100):
		properties = PoroelasticProperties(properties)
		self.load = settings["Parameters"]["Load"]["Value"]
		self.height = settings["Parameters"]["Dimensions"]["Height"]["Value"]
		self.dt = settings["Simulation"]["Timestep Size"]["Value"]
		self.c = (properties.k*properties.Q*properties.M)/(properties.mu*(properties.M + properties.alpha**2*properties.Q))
		self.M = properties.M
		self.alpha = properties.alpha
		self.Q = properties.Q
		self.Mu = properties.M + properties.alpha**2*properties.Q
		self.p0 = -(properties.alpha*properties.Q*self.load)/(properties.M + properties.alpha**2*properties.Q)
		self.numOfTerms = numOfTerms

	def getTime(self, step):
		return step*self.dt

	def exactPressureSolution(self, position, time):
		if near(time, 0.):
			return self.p0
		else:
			summation = 0
			pi = math.pi
			for j in range(1, self.numOfTerms):
				serial = ((2*j - 1)*pi)/(2.*self.height)
				term1 = (-1)**(j + 1)/(2*j - 1)
				term2 = math.exp(-self.c*time*serial**2)
				term3 = math.cos(((2*j - 1)*pi)/(2.*self.height)*position)
				summation += term1*term2*term3
			return summation*(4./pi)*self.p0

	def exactDisplacementSolution(self, position, time):
		if near(time, 0.):
			return self.load/(self.Mu)*position
		else:
			summation = 0
			pi = math.pi
			for j in range(1, self.numOfTerms):
				serial = ((2*j - 1)*pi)/(2.*self.height)
				term1 = (-1)**(j + 1)/(2*j - 1)**2
				term2 = math.exp(-self.c*time*serial**2)
				term3 = math.sin(((2*j - 1)*pi)/(2.*self.height)*position)
				summation += term1*term2*term3
			return self.load/(self.M)*position - ((8.*self.alpha**2*self.Q*self.height*self.load)/(pi**2*self.M*self.Mu))*summation