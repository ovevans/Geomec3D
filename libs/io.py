from dolfin import near, XDMFFile
import h5py
import json
import xmltodict


def getJsonData(jsonFile):
	f = open(jsonFile)
	data = json.load(f)
	f.close()
	return data


def saveJsonData(data, destination, filename):
	file = "{}/{}.json".format(destination, filename)
	output = open(file, "w+")
	output.write(json.dumps(data, indent="\t", separators=(",", ": ")))
	output.close()
	

def copyProperties(source, filename, destination, regionNames):
	properties = getJsonData("{}/{}".format(source, filename))
	file = "{}/{}".format(destination, filename)
	dataset = {}
	for region in regionNames:
		dataset[region] = properties[region]
	output = open(file, "w+")
	output.write(json.dumps(dataset, indent="\t",  separators=(',', ": ")))
	output.close()


class XDMFWriter(object):
	def __init__(self, directory, file):
		self.outputFile = XDMFFile("{}/{}.xdmf".format(directory, file))
		self.outputFile.parameters["rewrite_function_mesh"] = False
		self.outputFile.parameters["functions_share_mesh"] = True

	def writeSingle(self, solution, time=0.0):
		self.outputFile.write(solution, time)

	def writeMultiple(self, solutionList, time=0.0):
		for solution in solutionList:
			self.outputFile.write(solution, time)

	def close(self):
		self.outputFile.close()


class HDF5Reader(object):
	def __init__(self, directory, file):
		self.file = file
		self.h5 = h5py.File("{}/{}.h5".format(directory, file), "r")
		self.xdmf = xmltodict.parse(open("{}/{}.xdmf".format(directory, file)).read())

	def getCoordinates(self):
		return self.h5["Mesh"]["0"]["mesh"]["geometry"][()]

	def getElements(self):
		return self.h5["Mesh"]["0"]["mesh"]["topology"][()]

	def getNumOfSteps(self):
		return len(self.xdmf["Xdmf"]["Domain"]["Grid"]["Grid"])

	def getNode(self, point):
		x0 = point.x()
		y0 = point.y()
		z0 = point.z()
		coordinates = self.getCoordinates()
		for row in range(len(coordinates)):
			x = coordinates[row][0]
			y = coordinates[row][1]
			z = coordinates[row][2]
			if near(x, x0) and near(y, y0) and near(z, z0):
				return row

	def getNodesAlongAxis(self, axis, axis0=0.0):
		nodes = []
		coordinates = self.getCoordinates()
		for j, row in enumerate(coordinates):
			appendRow = True
			for i, coord in enumerate(row):
				if i != axis and near(coord, axis0):
					appendRow = appendRow and True
				elif i != axis and not near(coord, axis0):
					appendRow = appendRow and False
			if appendRow:
				nodes.append(j)
		return nodes

	def getNodeCoordinate(self, node, axis):
		coordinates = self.getCoordinates()
		return coordinates[node][axis]

	def getTimeLevel(self, step):
		return self.xdmf["Xdmf"]["Domain"]["Grid"]["Grid"][step]["Time"]["@Value"]

	def getSolutionAtStep(self, name, step):
		for orderdict in self.xdmf["Xdmf"]["Domain"]["Grid"]["Grid"][step]["Attribute"]:
			if orderdict["@Name"] == name:
				position = orderdict["DataItem"]["#text"].replace(self.file + ".h5:/VisualisationVector/", "")
				solution = self.h5["VisualisationVector"][position][()]
				return solution

	def getSolutionAtNodeAndStep(self, name, node, step):
		return self.getSolutionAtStep(name, step)[node]