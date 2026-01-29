import sympy as sp
from sympy.physics.mechanics import dynamicsymbols

# calculate M_Lambd and F_Lambd

t = dynamicsymbols._t

x, y, z, theta, phi = dynamicsymbols("x y z theta phi")  # x and y -> buoy coordinates

m_b, m_d, l, g = sp.symbols(
    "m_b m_d l g", positive=True
)  # mass buoy, mass drogue in [kg], wire length in [m], g in [m(s^2)]

k_b, k_d = sp.symbols("k_b k_d", positive=True)  # drag coefficients for buoy and drogue

U_b, V_b, U_d, V_d = sp.symbols(
    "U_b V_b U_d V_d", real=True
)  # surface horizontal velocities

# U_0, V_0 = sp.symbols('U_0 V_0', real = True) # surface horizontal velocities

r_b = sp.Matrix([x, y, 0])  # buoy position

r = l * sp.Matrix(
    [
        sp.sin(theta) * sp.cos(phi),
        sp.sin(theta) * sp.sin(phi),
        sp.cos(theta),
    ]
)

r_d = r_b + r  # drogue position

# velocities

v_b = r_b.diff(t)
v_d = r_d.diff(t)

u_b = sp.Matrix([U_b, V_b, 0])
u_d = sp.Matrix([U_d, V_d, 0])

def _mag(vec):
    return sp.sqrt(vec.dot(vec))


F_b = -k_b * _mag(v_b - u_b) * (v_b - u_b)  # drag buoy with ext. velocity
F_d = -k_d * _mag(v_d - u_d) * (v_d - u_d)  # drag drogue with ext. velocity

# lagrangian

T = sp.Rational(1, 2) * m_b * v_b.dot(v_b) + sp.Rational(1, 2) * m_d * v_d.dot(
    v_d
)  # kinetic energy

V = m_d * g * r_d[2]  # potential energy

L = T - V  # lagrangian

q = sp.Matrix([x, y, theta, phi])
qd = q.diff(t)
qdd = qd.diff(t)

Q = sp.Matrix([r_b.diff(qi).dot(F_b) + r_d.diff(qi).dot(F_d) for qi in q])

Q = sp.simplify(Q)

eoms = sp.Matrix(
    [L.diff(qdj).diff(t) - L.diff(qj) - Qj for qj, qdj, Qj in zip(q, qd, Q)]
)
eoms = sp.simplify(eoms)

M, F = sp.simplify(sp.linear_eq_to_matrix(eoms, list(qdd)))

xd, yd, thetad, phid = qd

args = (
    t, x, y, theta, phi, 
    xd, yd, thetad, phid,
    m_b, m_d, l, g, k_b, k_d,
    U_b, V_b, U_d, V_d
)

__all__ = ["M", "F", "args"]