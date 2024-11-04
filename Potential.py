import numpy as np
import matplotlib.pyplot as plt

class Potential:
    def __init__(self, N:int=100, a:int=10, m:float=1, h:float=1):
        self.N = N
        self.a = a
        self.m = m
        self.h = h
        self.x = np.linspace(-a, a, N)
        self.dx = self.x[1] - self.x[0]


    class Squared_Barrier:
        def __init__(self,parent, V0: float = 10, width: float = 0.2,
                        V_min: float = 0):
            self.parent = parent
            self.name="Squared Barrier Potential"
            self.V0 = V0
            self.width = width
            self.V_min = V_min
            
            
            if not 0 < width < 1:
                raise ValueError("Invalid width. Width must be between 0 and 1 (exclusive).")
            
            
        def get_potential(self,n: int = 1,toll=1e-34):
            if n <= 0 or n>=self.parent.N:
                raise ValueError("Invalid number of barriers. Number of barriers must be a positive integer and less than N.")
            V = np.full(self.parent.N, self.V_min, dtype=float)
            width_barrier= 2*self.width*self.parent.a
            width_sub_barrier= width_barrier/(2*n-1)
            start = -width_barrier/2
            for j in range(n):
                center = start + j * width_sub_barrier*2
                for i in range(self.parent.N):
                    if abs(self.parent.x[i]-center-width_sub_barrier/2)<width_sub_barrier/2:
                        V[i]=self.V0 if abs(self.V0)>toll else toll
            return V
            
        def get_k(self,E,n: int = 1):
            return np.sqrt(2*self.parent.m*(E - self.get_potential(n)) / self.parent.h**2, dtype=complex)
        
    class Step_Barrier:
        def __init__(self,parent, V0: float=10,V_min=0,offset:float=0):
            self.parent = parent
            self.name = "Step Barrier Potential"
            self.V0 = V0
            self.V_min = V_min
            self.offset = offset

        
        def get_potential(self,n:int=1,):
            if n <= 0 or n >= self.parent.N:
                raise ValueError("Invalid number of barriers. Number of barriers must be a positive integer and less than N.")
            n=n+1
            V = np.full(self.parent.N, self.V_min)
            step_heights = np.linspace(self.V_min,self.V0,n)
            step_positions = np.linspace(-self.parent.a/2,self.parent.a/4,n)+self.offset

            for i, height in enumerate(step_heights):
                if i < len(step_positions) - 1:
                    x_start = step_positions[i]
                    x_end = step_positions[i + 1]
                    for j in range(self.parent.N):
                        if x_start <= self.parent.x[j] < x_end:
                            V[j] = height
                else:
                    x_start = step_positions[i]
                    for j in range(self.parent.N):
                        if self.parent.x[j] >= x_start:
                            V[j] = height
            return V
        def get_k(self,E,n:int=1):
            k = np.sqrt(2*self.parent.m*(E - self.get_potential(n)) / self.parent.h**2, dtype=complex)
            return k
        

    def get_plot_VE(self,V,E,ax=plt):
        E =np.ones(self.N)*E
        ax.plot(self.x,V, label="V(x)")
        ax.plot(self.x,E, label = "Energy",linestyle="--")
        ax.xlabel('x')
        ax.ylabel('V(x)')
        ax.title(f'Potential of the System')
        ax.grid(False)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
    def get_plot_k(self,V,E,ax='plt',ax1='plt.'):
        E =np.ones(self.N)*E
        k = V._get_k(E)
        ax.plot(self.x,np.imag(k), label = "k(x) imag", linestyle="-.")
        ax1.xlabel('x')
        ax1.ylabel('V(x)')
        ax1.title(f'Potential of the System')
        ax1.grid(False)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.show()
        
        # def Squared_Barrier(self, V0: float = 10, width: float = 0.2,
        #                     V_min: float = 0, n: int = 1,toll=1e-34, name="Squared Barrier Potential"):
        #     self.name= "Squared_Barrier"
        #     if not 0 < width < 1:
        #         raise ValueError("Invalid width. Width must be between 0 and 1 (exclusive).")
        #     if n <= 0 or n>=self.N:
        #         raise ValueError("Invalid number of barriers. Number of barriers must be a positive integer and less than N.")
        #
        #     V = np.full(self.N, V_min, dtype=float)
        #     width_barrier= 2*width*self.a
        #     width_sub_barrier= width_barrier/(2*n-1)
        #     start = -width_barrier/2
        #     for j in range(n):
        #        center = start + j * width_sub_barrier*2
        #        for i in range(self.N):
        #         if abs(self.x[i]-center-width_sub_barrier/2)<width_sub_barrier/2:
        #            V[i]=V0 if abs(V0)>toll else toll
        #     return V
        #
        # def Step_Barrier(self, V0: float=10,V_min=0,n:int=1,offset:float=0) -> np.ndarray:
        #     self.Step_Barrier.__name__ = "Step Barrier Potential"
        #
        #     if n<=0 or n>=self.N:
        #         raise ValueError("Invalid number of barriers. Number of barriers must be a positive integer and less than N.")
        #
        #     n=n+1
        #     V = np.full(self.N, V_min)
        #     step_heights = np.linspace(V_min,V0,n)
        #     step_positions = np.linspace(-self.a/2,self.a/4,n)+offset
        #
        #     for i, height in enumerate(step_heights):
        #       if i < len(step_positions) - 1:
        #           x_start = step_positions[i]
        #           x_end = step_positions[i + 1]
        #           for j in range(self.N):
        #               if x_start <= self.x[j] < x_end:
        #                   V[j] = height
        #       else:
        #           x_start = step_positions[i]
        #           for j in range(self.N):
        #               if self.x[j] >= x_start:
        #                   V[j] = height
        #     return V
        #
        # def MorseFebnash_Potential(self,V0:float=10, mu=0.2, xe=0,offset=0):
    #  pass


