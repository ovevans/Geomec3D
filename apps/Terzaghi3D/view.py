import sys
sys.path.insert(0, '../..')

from dolfin import Point
import matplotlib.pyplot as plt
import numpy as np

from libs.io import *
from libs.exact import *


""" INPUT """

directory1 = "results/P1P1"
directory2 = "results/P2P1"
label1 = "P1P1"
label2 = "P2P1"
mark1 = "o"
mark2 = "x"
mark3 = "s"
propertiesFile = "poroelastic_properties.json"
resultsFile = "results"
settingsFile = "settings.json"

""" START """

# Import data
properties = getJsonData("{}/{}".format(directory1, propertiesFile))
settings = getJsonData("{}/{}".format(directory1, settingsFile))
for key in properties:
	medium = key
properties = properties[key]
reader1 = HDF5Reader(directory1, resultsFile)
reader2 = HDF5Reader(directory2, resultsFile)
# Exact solution
terza = Terzaghi(properties, settings, numOfTerms=100)
# Arrange data
nodes = reader1.getNodesAlongAxis(2)
numOfSteps = reader1.getNumOfSteps()
plottedSteps = [0,
				int(numOfSteps/128),
				int(numOfSteps/8),
				int(numOfSteps/2),
				int(numOfSteps-1)]
# Plot data
fig = plt.figure(figsize=(8, 5))
fig.subplots_adjust(top=0.88, bottom=0.15, left=0.08, right=0.95, wspace=0.2)
z1 = [float(reader1.getNodeCoordinate(node, 2)) for node in nodes]
z2 = [float(reader2.getNodeCoordinate(node, 2)) for node in nodes]
zmin = min(z1)
zmax = max(z1)
zExact = np.linspace(zmin, zmax, 1000)
fig.add_subplot(1, 2, 1)
for i in range(len(plottedSteps)):
	step = plottedSteps[i]
	tExact = terza.getTime(step)
	p1 = [float(reader1.getSolutionAtNodeAndStep("p", node, step))/1e3 for node in nodes]
	p2 = [float(reader2.getSolutionAtNodeAndStep("p", node, step))/1e3 for node in nodes]
	if i == 0:
		plt.plot([terza.exactPressureSolution(z, tExact)/1e3 for z in zExact], zExact, "-", color="grey", label="Exact")
		plt.plot(p1, z1, mark1, fillstyle="none", label=label1, ms=6, mec="k", mew=0.75)
		plt.plot(p2, z2, mark2, fillstyle="none", label=label2, ms=6, mec="k", mew=0.75)
	else:
		plt.plot([terza.exactPressureSolution(z, tExact)/1e3 for z in zExact], zExact, "-", color="grey")
		plt.plot(p1, z1, mark1, fillstyle="none", ms=6, mec="k", mew=0.75)
		plt.plot(p2, z2, mark2, fillstyle="none", ms=6, mec="k", mew=0.75)
plt.xlabel('Pressure (kPa)')
plt.ylabel('Height (m)')
plt.grid(which='major', axis='both')
fig.add_subplot(1, 2, 2)
for i in range(len(plottedSteps)):
	step = plottedSteps[i]
	tExact = terza.getTime(step)
	w1 = [reader1.getSolutionAtNodeAndStep("u", node, step)[2]*1e3 for node in nodes]
	w2 = [reader2.getSolutionAtNodeAndStep("u", node, step)[2]*1e3 for node in nodes]
	plt.plot([terza.exactDisplacementSolution(z, tExact)*1e3 for z in zExact], zExact, "-", color="grey")
	plt.plot(w1, z1, mark1, fillstyle="none", ms=6, mec="k", mew=0.75)
	plt.plot(w2, z2, mark2, fillstyle="none", ms=6, mec="k", mew=0.75)
plt.xlabel('Vertical Displacement (mm)')
plt.ylabel('Height (m)')
plt.grid(which='major', axis='both')
fig.legend(loc='upper center', ncol=4)
plt.show()