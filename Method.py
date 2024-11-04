from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt

class Analytics:
    def __init__(self,V,k,a,potential:str):
        self.V = V
        self.k = k
        self.N = len(V)
        self.potential = potential
        self.a = a

    def transmission_relfection_coefficients(self,energy):
        k_dictionary = {f'k_{i}': [key, len(list(group))] for i, (key, group) in enumerate(groupby(self.k))}
        for j, value in enumerate(k_dictionary.values()): globals()[f'k_{j}'] = value
        k_0 = k_dictionary.get('k_0', [None, None])
        V_dictionary = {f'V_{i}': [key, len(list(group))] for i, (key, group) in enumerate(groupby(self.V))}
        for j, value in enumerate(V_dictionary.values()): globals()[f'V_{j}'] = value
        
        L = V_dictionary['V_1'][1]/self.N
        V0 = V_dictionary['V_1'][0]
        if self.potential=="Squared_Barrier":
            k = k_1[0]*1j if energy < V0 else k_0[0]
            if energy < V0:
                T = 1 / (1 + (V0**2 * np.sinh(k * self.a)**2) / (4 * energy * (V0 - energy)))
            else:
                T = 1 / (1 + (V0**2 * np.sin(k * L)**2) / (4 * energy * (energy - V0)))
            
        elif self.potential=="Step_Barrier":
            k1 = k_0[0]
            k2 = k_1[0]
            if energy < V0:
                T = 0
            else:
                T = (k2 * k1) / (k1 + k2)**2
                
        elif self.potential=="MorseFebnash_Potential":
            #WKB approximation
            x1, x2 = xe - 0.5e-10, xe + 0.5e-10  # Guess around xe
            # Calculate the WKB integral
            gamma, _ = quad(integrand, x1, x2)
            # Calculate transmission probability
            T = np.exp(-2 * gamma)
   
        T= T**2
        R = (1-T)
        return T,R
    def plot(self,energy):
        T,R = self.transmission_relfection_coefficients(energy)
        plt.plot(energy,T,label="Transmission")
        plt.plot(energy,R,label="Reflection")
        plt.xlabel("Energy")
        plt.ylabel("Probability")
        plt.legend()
        plt.show()
    
    def TE_var(self,energy):
        T_list, R_list = [], []
        for E in energy:
            T, R = self.transmission_relfection_coefficients(E)
            T_list.append(T)
            R_list.append(R)
        return np.array(T_list), np.array(R_list)
    
import numpy as np
from itertools import groupby

class MatrixTransfer:
    def __init__(self, k, n):
        self.k = k
        self.n = n
        self.k_dictionary = {f'k_{i}': [key, len(list(group))] for i, (key, group) in enumerate(groupby(k))}
        for j, value in enumerate(self.k_dictionary.values()): 
            globals()[f'k_{j}'] = value
        K1, A1 = (np.array(list(self.k_dictionary.values())).T)[0], (np.array(list(self.k_dictionary.values())).T).real[1]
        A1 = 1 / (len(k)) * A1
        self.K, self.A = np.array(K1), np.array(A1)
        self.k1 = self.K[0]
        self.k2 = self.K[1]
        self.a = self.A[1]
        self.M_total = np.identity(2, dtype=complex)
        self.Ai, self.Bi = [1+0j], [0j]
        self.psi = []

    def transfer_matrix(self, k1, k2, a):
        M11 = 0.5 * (1 + k2 / k1) * np.exp(1j * (k1 - k2) * a)
        M12 = 0.5 * (1 - k2 / k1) * np.exp(1j * (k1 + k2) * a)
        M21 = 0.5 * (1 - k2 / k1) * np.exp(-1j * (k1 + k2) * a)
        M22 = 0.5 * (1 + k2 / k1) * np.exp(-1j * (k1 - k2) * a)
        return np.array([[M11, M12], [M21, M22]])

    def calculate(self):
        for i in range(2 * self.n):
            k1 = self.K[i]
            k2 = self.K[i + 1]
            a = self.A[i]
            M = self.transfer_matrix(k1, k2, a)
            self.Ai.append(M[0, 0])
            self.Bi.append(M[0, 1])
            self.M_total = self.M_total @ M

        R = abs((self.M_total[1, 0] / self.M_total[0, 0]))**2
        T = abs((1 / self.M_total[0, 0]))**2
        return T, R
    
    def TE_var(self, E_var,potential, n: int = 1):
        T_list, R_list = [], []
        for E in E_var:
            k0 = potential.get_k(E, n)
            Method = MatrixTransfer(k0, n)
            T, R = Method.calculate()
            T_list.append(T)
            R_list.append(R)
        return np.array(T_list), np.array(R_list)
    

from numpy import ones, imag, pi, matrix, array
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import inv as inverse
from itertools import groupby
from typing import List, Tuple, Any

class GreenFunction_V3:
    def __init__(self, m: float, h: float, E: float):
        
        self.__name__ = "Green's Function Method"
        if m <= 0 or h <= 0:
            raise ValueError("Mass and Planck's constant must be positive")
        self.m = m
        self.h = h 
        self.E = E
        self.t = self.h**2/(2*self.m)

    def Hamiltonian(self, V: Any, n: int = 1) -> diags:
        try:
            potential = V.get_potential(n)
            N = V.parent.N
            dx = V.parent.x[1] - V.parent.x[0]
            self.t = self.t/dx**2

            potential_dictionary = {
                f'V_{i}': [key, len(list(group))] 
                for i, (key, group) in enumerate(groupby(potential))
            }
            
            for j, value in enumerate(potential_dictionary.values()):
                globals()[f'V_{j}'] = value
                
            V_list, A_list = (array(list(potential_dictionary.values())).T)[0], \
                            (array(list(potential_dictionary.values())).T).real[1]

            t = self.t * ones(N+1) 
            return diags([-t[:-1], potential+2*t[:-1], -t[:-1]], [-1, 0, 1], shape=(N,N))
            
        except Exception as e:
            raise ValueError(f"Error constructing Hamiltonian: {str(e)}")

    def GreenFunction_for_finding_Self_Energy(self, V: Any, Green_surfaceL: float = 0, Green_surfaceR: float = 0) -> matrix:
        """Calculate Green's function for self-energy."""
        self.Green_surfaceL = Green_surfaceL
        self.Green_surfaceR = Green_surfaceR
        N = V.parent.N
        dx = V.parent.x[1] - V.parent.x[0]
        self.t = self.t/dx
        
        H = self.Hamiltonian(V)
        Sigma = eye(N) * 0
        Sigma = Sigma.tocsc()
        Sigma[0,0] = self.t * Green_surfaceL * self.t
        Sigma[-1,-1] = self.t * Green_surfaceR * self.t
        
        return inverse(self.E * eye(N) - H - Sigma)

    def GreenFunction_for_finding_center_green_function(self, V: Any) -> Tuple:
        """Calculate center Green's function."""
        GreenFunction_0 = self.GreenFunction_for_finding_Self_Energy(V, 0, 0)
        gL, gR = GreenFunction_0[0,0], GreenFunction_0[-1,-1]
        Gc = self.GreenFunction_for_finding_Self_Energy(V, gL, gR)
        Gc = Gc[Gc.shape[0] // 2, Gc.shape[1] // 2]
        return Gc, gL, gR

    def DensityOfState_in_center(self, V: Any) -> float:
        """Calculate density of states at the center."""
        G, _, _ = self.GreenFunction_for_finding_center_green_function(V)
        return -1/pi * imag(G)

    def GreenFunction(self, V: Any) -> matrix:
        """Calculate full Green's function."""
        Gc, gL, gR = self.GreenFunction_for_finding_center_green_function(V)
        GL = gL * (1 + self.t * Gc * self.t * gL)
        GCL = gL * self.t * Gc
        GRL = gL * self.t * Gc * self.t * gR
        GLC = Gc * self.t * gL
        GC = Gc
        GRC = Gc * self.t * gR
        GLR = gR * self.t * Gc * self.t * gL
        GCR = gR * self.t * Gc
        GR = gR * (1 + self.t * Gc * self.t * gR)
        
        return matrix([[GL, GCL, GRL],
                      [GLC, GC, GRC],
                      [GLR, GCR, GR]])

    
    def TransmissionCoefficient(self, V: Any) -> float:
        
        
        Gc, gL, gR = self.GreenFunction_for_finding_center_green_function(V)
        GammaL = 1j * (gL - np.conj(gL))
        GammaR = 1j * (gR - np.conj(gR))
        T = np.real(GammaL * Gc * GammaR * np.conj(Gc))
        return T
    