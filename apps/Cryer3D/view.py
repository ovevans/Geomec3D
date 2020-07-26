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
propertiesFile = "poroelastic_properties.json"
resultsFile = "results"
settingsFile = "settings.json"

""" START """

# Import data
properties = getJsonData("{}/{}".format(directory1, propertiesFile))
settings = getJsonData("{}/{}".format(directory1, settingsFile))
for key in properties:
	medium = key
properties = properties[medium]
reader1 = HDF5Reader(directory1, resultsFile)
reader2 = HDF5Reader(directory2, resultsFile)
# Exact solution
cryer = Cryer(properties, settings, numOfRoots=100)
# Arrange data
t1 = []
t2 = []
p1 = []
p2 = []
step = 0
numOfSteps = reader1.getNumOfSteps()
while step < numOfSteps:
	t1.append(float(reader1.getTimeLevel(step))/3.6e3)
	t2.append(float(reader2.getTimeLevel(step))/3.6e3)
	p1.append(float(reader1.getSolutionAtNodeAndStep("p", reader1.getNode(Point(0., 0., 0.)), step))/1e3)
	p2.append(float(reader2.getSolutionAtNodeAndStep("p", reader2.getNode(Point(0., 0., 0.)), step))/1e3)
	step += 6
tmin = min(t1)
tmax = max(t1)
tExact = np.linspace(tmin, tmax, 1000)
# Plot data
fig = plt.figure(figsize=(8, 5))
plt.plot(tExact, [cryer.exactSolution(t*3.6e3)/1e3 for t in tExact], "-", color="grey", label="Exact")
plt.plot(t1, p1, mark1, fillstyle="none", label=label1, ms=6, mec="k", mew=0.75)
plt.plot(t2, p2, mark2, fillstyle="none", label=label2, ms=6, mec="k", mew=0.75)
plt.xlabel('Time (hours)')
plt.ylabel('Pressure (kPa)')
plt.grid(which='major', axis='both')
fig.legend(loc='upper center', ncol=4)
plt.show()