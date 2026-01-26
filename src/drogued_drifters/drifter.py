class DroguedDrifter:

    def __init__(self, k_b: float = None, k_d: float = None):
        # DroguedDrifter attributes hold physics parameters (masses, drag coeffs, ...)
        self.k_b = k_b
        self.k_d = k_d

        self.M_lbd = M_lbd
        self.F_lbd = F_lbd

    def solve_sp_MF(self):
        raise NotImplementedError

    def get_full_solution(self):
        # TODO: this method gets initial conditions and runtime etc.
        raise NotImplementedError

    def get_netto_uv(self):
        # TODO: this method gets initial conditions and runtime etc.
        # TODO: this method gets U,V profile
        # TODO: Calls .get_full_solution() and extracts netto equilibrium drift
        raise NotImplementedError