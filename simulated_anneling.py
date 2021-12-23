import random
import math
from imageprocessing import kmeans

class anneling(): 

    def __init__(self,_T):
        
        self.T=_T
        self.t=len(_T)
        self.it=0

    def next_state(self):

        """ Genera un estado vecino"""

        aux=self.actual_state.copy()
        index=random.sample(range(self.d2),2)
        aux[index[0]],aux[index[1]] = self.actual_state[index[1]],self.actual_state[index[0]]
        
        return aux

    def energy(self,state):
        
        """Devuelve la energia del conjunto de ordenes o el camino"""

        dT=kmeans(self.x_T,self.y_T,state,kn=self.kn,anelling=True)
        energy=dT.sum()

        return energy

    def probability(self,delta_energy,Temp):
        
        """ Devuelve una probabilidad en función de la temperatura"""

        return math.exp(delta_energy/Temp)

    def search(self,cluster_in,x_T,y_T,kn):

        it=0
        self.actual_state=cluster_in
        self.x_T=x_T
        self.y_T=y_T
        self.kn=kn
        self.list_energy=[]
        self.next_stateaux=[]
        self.d2=len(cluster_in)
        self.E1=self.energy(self.actual_state.copy())

        for i in range(self.t):

            it+=1

            if abs(self.T[i])==0 or it==self.it :

                return self.actual_state,self.list_energy
                    
                        
            self.next_stateaux=self.next_state()
            E2=self.energy(self.next_stateaux.copy())
            deltaenergy=self.E1-E2

            if deltaenergy>0:

                it=0
                self.actual_state=self.next_stateaux
                self.E1=E2

            else:

                prob=self.probability(deltaenergy,self.T[i])
                choise=random.choices([0,1],[1-prob,prob])

                if (choise[0]==1): 

                    self.actual_state=self.next_stateaux
                    self.E1=E2

            self.list_energy.append(self.E1)


def ley_enfriamiento(tem_max,len_enfria,coef_exp):

    T=[tem_max]

    for i in range(len_enfria):

        T.append(T[-1]/coef_exp)

    T[-1]=0

    return T


    



    