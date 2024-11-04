import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby

from Potential import Potential
from Method import Analytics, MatrixTransfer

from scipy.constants import physical_constants as pc


# Constants
N=100
L=1
m,h=1,1
V0=10
E=6
n=1
E_var = np.linspace(0,15,N)

potential = Potential(N,L,m,h)
Squared_Barrier = potential.Squared_Barrier(potential,V0=V0,width=0.25,V_min=0)
Step_Barrier=potential.Step_Barrier(potential,V0=V0,V_min=0)

V_Squared_Barrier = Squared_Barrier.get_potential(n)
V_Step_Barrier = Step_Barrier.get_potential(n)
x,dx = potential.x,potential.dx
k_Squared_Barrier = Squared_Barrier.get_k(E,n)
k_Step_Barrier=Step_Barrier.get_k(E,n)

# Analytical
analytical_Squared_Barrier = Analytics(V=V_Squared_Barrier,k=k_Squared_Barrier,a=L,potential="Squared_Barrier")
analytical_Step_Barrier= Analytics(V=V_Step_Barrier,k=k_Step_Barrier,a=L,potential="Step_Barrier")

R_analytical_Squared_Barrier,T_analytical_Squared_Barrier=analytical_Squared_Barrier.TE_var(E_var)
R_analytical_Step_Barrier,T_analytical_Step_Barrier=analytical_Step_Barrier.TE_var(E_var)


#Tranfer Matrix
matrixTransfer_Squared_Barrier = MatrixTransfer(k_Squared_Barrier,n)
matrixTransfer_Step_Barrier = MatrixTransfer(k_Step_Barrier,n)

T_transfer_Squared_Barrier,R_transfer_Squared_Barrier=matrixTransfer_Squared_Barrier.TE_var(E_var,Squared_Barrier,n)
#T_transfer_Step_Barrier,R_transfer_Step_Barrier=matrixTransfer_Step_Barrier.TE_var(E_var,Step_Barrier,n)



# Plot
potential.get_plot_VE(V_Squared_Barrier,E)
potential.get_plot_VE(V_Step_Barrier,E)
plt.plot(E_var,T_analytical_Squared_Barrier,label="T_analytical_Squared_Barrier")
plt.plot(E_var,R_analytical_Squared_Barrier,label="R_analytical_Squared_Barrier")
plt.xlabel("E")
plt.ylabel("T/R")
plt.legend()
plt.grid(True)
plt.show()
plt.plot(E_var,T_analytical_Step_Barrier,label="T_analytical_Step_Barrier")
plt.plot(E_var,R_analytical_Step_Barrier,label="R_analytical_Step_Barrier")
plt.xlabel("E")
plt.ylabel("T/R")
plt.legend()
plt.grid(True)
plt.show()
plt.plot(E_var,T_transfer_Squared_Barrier,label="T_transfer_Squared_Barrier")
plt.plot(E_var,R_transfer_Squared_Barrier,label="R_transfer_Squared_Barrier")
plt.xlabel("E")
plt.ylabel("T/R")
plt.legend()
