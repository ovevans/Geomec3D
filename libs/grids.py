from dolfin import near, Point, SubDomain, Mesh, MeshFunction, Measure, RectangleMesh, BoxMesh
from mshr import Sphere, Box, generate_mesh


class BoxGrid(object):
	def __init__(self, x0, x1, y0, y1, z0, z1, nx, ny, nz):
		class Left(SubDomain):
			def inside(self, x, on_boundary):
				return near(x[0], x0)
		class Right(SubDomain):
			def inside(self, x, on_boundary):
				return near(x[0], x1)
		class Back(SubDomain):
			def inside(self, x, on_boundary):
				return near(x[1], y0)
		class Front(SubDomain):
			def inside(self, x, on_boundary):
				return near(x[1], y1)
		class Bottom(SubDomain):
			def inside(self, x, on_boundary):
				return near(x[2], z0)
		class Top(SubDomain):
			def inside(self, x, on_boundary):
				return near(x[2], z1)
		self.mesh = BoxMesh(Point(x0, y0, z0), Point(x1, y1, z1), nx, ny, nz)
		self.domains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
		self.domains.set_all(0)
		self.dx = Measure('dx', domain=self.mesh, subdomain_data=self.domains)
		self.boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
		self.boundaries.set_all(0)
		self.left = Left()
		self.left.mark(self.boundaries, 1)
		self.right = Right()
		self.right.mark(self.boundaries, 2)
		self.front = Front()
		self.front.mark(self.boundaries, 3)
		self.back = Back()
		self.back.mark(self.boundaries, 4)
		self.bottom = Bottom()
		self.bottom.mark(self.boundaries, 5)
		self.top = Top()
		self.top.mark(self.boundaries, 6)
		self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
		self.dS = Measure('dS', domain=self.mesh, subdomain_data=self.boundaries)


class QuarterSphereGrid(object):
	def __init__(self, x0, y0, z0, R, n):
		class Boundary(SubDomain):
			def inside(self, x, on_boundary):
				return on_boundary
		class X_Symmetric(SubDomain):
			def inside(self, x, on_boundary):
				return near(x[0], x0)
		class Y_Symmetric(SubDomain):
			def inside(self, x, on_boundary):
				return near(x[1], y0)
		class Z_Symmetric(SubDomain):
			def inside(self, x, on_boundary):
				return near(x[2], z0)
		self.mesh = generate_mesh(Sphere(Point(x0, y0, z0), R) - Box(Point(x0 + R, y0, z0 - R), Point(x0 - R, y0 - R, z0 + R)) - Box(Point(x0 - R, y0 + R, z0), Point(x0 + R, y0, z0 - R)) - Box(Point(x0, y0, z0), Point(x0 - R, y0 + R, z0 + R)), n)
		self.domains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
		self.domains.set_all(0)
		self.dx = Measure('dx', domain=self.mesh, subdomain_data=self.domains)
		self.boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
		self.boundaries.set_all(0)
		self.boundary = Boundary()
		self.boundary.mark(self.boundaries, 1)
		self.x_symmetric = X_Symmetric()
		self.x_symmetric.mark(self.boundaries, 2)
		self.y_symmetric = Y_Symmetric()
		self.y_symmetric.mark(self.boundaries, 3)
		self.z_symmetric = Z_Symmetric()
		self.z_symmetric.mark(self.boundaries, 4)
		self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)
		self.dS = Measure('dS', domain=self.mesh, subdomain_data=self.boundaries)