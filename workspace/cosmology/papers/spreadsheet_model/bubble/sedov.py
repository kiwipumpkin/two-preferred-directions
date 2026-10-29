"""Sedov-Taylor self-similar density G(xi) = rho/rho_1 behind a strong spherical blast (uniform ambient), gamma given."""
import numpy as np
from scipy.integrate import solve_ivp
def sedov_G(gamma, n=2000):
    d = 2.0 / 5.0
    def rhs(s, y):                       # s = ln xi ; y = V, lnG, lnP
        V, lG, lP = y; G = np.exp(lG); P = np.exp(lP)
        # G'/G = -(3V+V')/(V-1) ; P'/P = gamma G'/G - (2V - 2/d)/(V-1)
        # (V-1)V' + V^2 - V/d + P/G (2 + P'/P) = 0
        a = P / G; c = gamma / (V - 1)
        # P'/P = -c(3V+V') - (2V-2/d)/(V-1) = A0 + A1 V'
        A0 = -c * 3 * V - (2 * V - 2 / d) / (V - 1); A1 = -c
        Vp = -(V * V - V / d + a * (2 + A0)) / ((V - 1) + a * A1)
        Gp = -(3 * V + Vp) / (V - 1)
        return [Vp, Gp, A0 + A1 * Vp]
    y0 = [2 / (gamma + 1), np.log((gamma + 1) / (gamma - 1)), np.log(2 / (gamma + 1))]
    ev = lambda s, y: y[1] + 60; ev.terminal = True
    sol = solve_ivp(rhs, (0, np.log(1e-6)), y0, rtol=1e-10, atol=1e-12, dense_output=True, events=ev)
    s = np.linspace(sol.t[-1], 0, n); xi = np.exp(s); G = np.exp(sol.sol(s)[1])
    return xi, G
if __name__ == "__main__":
    for g in (5/3, 4/3):
        xi, G = sedov_G(g)
        m = np.trapezoid(4*np.pi*xi**2*G, xi) / (4*np.pi/3)
        print(g, "mass check (should be 1):", m, "G at xi=0.5,0.8,0.9,0.95,1:", [float(np.interp(x, xi, G)) for x in (0.5, 0.8, 0.9, 0.95, 1.0)])
