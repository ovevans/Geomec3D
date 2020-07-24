from dolfin import Mesh, MeshValueCollection, XDMFFile
from dolfin.cpp.mesh import MeshFunctionSizet, SubsetIterator


def DolfinReader(directory, file):
	msh = Mesh()
	with XDMFFile("{}/{}.xdmf".format(directory, file)) as infile:
		infile.read(msh)
		dim = msh.topology().dim()
	mvc = MeshValueCollection("size_t", msh, dim=dim-1)
	with XDMFFile("{}/{}_facets.xdmf".format(directory, file)) as infile:
		infile.read(mvc, "boundaries")
	boundaries = MeshFunctionSizet(msh, mvc)
	mvc = MeshValueCollection("size_t", msh, dim=dim)
	with XDMFFile("{}/{}_physical_region.xdmf".format(directory, file)) as infile:
		infile.read(mvc, "subdomains")
	subdomains = MeshFunctionSizet(msh, mvc)
	return msh, boundaries, subdomains


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