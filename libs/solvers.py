from multiphenics import block_solve


class Standard(object):
	def __init__(self, A, X, b):
		block_solve(A, X.block_vector(), b)
		self.solution = X


class Mumps(object):
	def __init__(self, A, X, b):
		block_solve(A, X.block_vector(), b, "mumps")
		self.solution = X