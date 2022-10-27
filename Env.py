"""
Projet MADI : environnement labyrinthe
Jérémy DUFOURMANTELLE
Nikola KOSTADINOVIC
"""

import numpy as np
from termcolor import colored

class EnvLabyrinthe() :

    def __init__(
        self,
        nblignes : int,
        nbcolonnes : int,
        alea : int,

    ) :
        self.nblignes = nblignes
        self.nbcolonnes = nbcolonnes

        # self.nb_states = nblignes*nbcolonnes
        self.nb_actions = 8 


        self.alea = alea
        self.costs = {
            0 : 0,
            1 : 2,
            2 : 4,
            3 : 8,
            4 : 16
        }

        # Initialisation des valeurs
        self.grid = np.zeros((nblignes,nbcolonnes), dtype=int)
        self.PosX = 0
        self.PosY = 0
        self.finalX = nblignes-1
        self.finalY = nbcolonnes-1

        self.init_grid()

        cpt_index = 0
        self.state_dict = dict()
        for i in range(nblignes):
            for j in range(nbcolonnes):
                if self.grid[i,j] >= 0 :
                    self.state_dict[(i,j)] = cpt_index
                    cpt_index += 1
        self.nb_states = cpt_index

    def get_index(self,i,j):
        return i * self.nbcolonnes + j

    def make_action(self,action,i,j):
        actions_values = ['f','g','h','j','y','u','r','t']
        index_action = actions_values[action]

        li = i
        cj = j

        new_li = li
        new_cj = cj
        reward = 0
        changed = 0

        # print(f"action a faire : {action} - case {i,j} - {self.grid[(i,j)]}")
        #input()

        # deplacement (-2,1)
        if index_action == 'y' and li>1 and cj < self.nbcolonnes-1 and self.grid[li-2,cj+1]>-1:
            new_li = li-2
            new_cj = cj+1
            reward = self.reward(new_li,new_cj)
            changed=1
        # deplacement (-2,-1)
        if index_action == 't' and li>1 and cj > 0 and self.grid[li-2,cj-1]>-1:
            new_li = li-2
            new_cj = cj-1
            reward = self.reward(new_li,new_cj)
            changed=1
        # deplacement (-1,2)
        if index_action == 'u' and li>0 and cj < self.nbcolonnes-2 and self.grid[li-1,cj+2]>-1:
            new_li = li-1
            new_cj = cj+2
            reward = self.reward(new_li,new_cj)
            changed=1
        # deplacement (-1,-2)
        if index_action == 'r' and li>0 and cj >1 and self.grid[li-1,cj-2]>-1:
            new_li = li-1
            new_cj = cj-2
            reward = self.reward(new_li,new_cj)
            changed=1
        # deplacement (2,1)  
        if index_action == 'h' and li<self.nblignes-2 and cj < self.nbcolonnes-1 and self.grid[li+2,cj+1]>-1:
            new_li = li+2
            new_cj = cj+1
            reward = self.reward(new_li,new_cj)
            changed=1
        # deplacement (2,-1)
        if index_action == 'g' and li<self.nblignes-2 and cj > 0 and self.grid[li+2,cj-1]>-1:
            new_li = li+2
            new_cj = cj-1
            reward = self.reward(new_li,new_cj)
            changed=1
        # deplacement (1,2)
        if index_action == 'j' and li<self.nblignes-1 and cj < self.nbcolonnes-2 and self.grid[li+1,cj+2]>-1:
            new_li = li+1
            new_cj = cj+2
            reward = self.reward(new_li,new_cj)
            changed=1
        # deplacement (1,-2)
        if index_action == 'f' and li<self.nblignes-1 and cj >1 and self.grid[li+1,cj-2]>-1:
            new_li= li+1
            new_cj = cj-2
            reward = self.reward(new_li,new_cj)
            changed=1
        
        return reward,new_li,new_cj,changed

    def step(
        self,
        action : int
    ) :
        
        reward,self.PosX,self.PosY,changed = self.make_action(action,self.PosX,self.PosY)
        
        # La variable alea =1 si on veut des effets aleatoires sinon les transitions sont deterministes
        #On ajoute un effet aleatoire dans les transitions
        if self.alea==1 and changed==1:
            t=np.random.uniform(0,1)    
            if t>0.5:
                d=np.random.randint(8)
                dli=0
                if d== 0 or d==1 or d==2:
                    dli=-1
                if d== 4 or d==5 or d==6:
                    dli==1
                dcj=0
                if d==0 or d==7 or d==6:
                    dcj=-1
                if d==2 or d==3 or d==4:
                    dcj=1    
            # l'effet aleatoire est applique s'il cree un deplacement sur une case admissible     
                NewPosY = self.PosY+dli
                NewPosX = self.PosX+dcj        
                #newcj=int((NewPosX-30)/(20*zoom))
                #newli=int((NewPosY-30)/(20*zoom))   
                if NewPosX>=0 and NewPosY>=0 and NewPosX<=self.nblignes-1 and NewPosY<=self.nbcolonnes-1 and self.grid[NewPosX,NewPosY]>-1:
                    self.PosY=NewPosY
                    self.PosX=NewPosX

        return self.state_dict[(self.PosX,self.PosY)],reward,self.isFinal()

    def isFinal(self):
        return self.PosX == self.finalX and self.PosY == self.finalY

    def isFinalcoords(self,i,j):
        return i == self.finalX and j == self.finalY

    def reset(self):
        pass

    def render(self) :
        colors = {
            0 : "white",
            1 : "green",
            2 : "blue",
            3 : "red",
            -1 : "grey",
            4 : "magenta"

        }
        """
        Affichage du labyrinthe dans le terminal
        """
        for i in range(self.nblignes):
            for j in range(self.nbcolonnes):
                if i == self.PosX and j == self.PosY :
                    print(colored("J","yellow"), end=" ")
                else :
                    if self.grid[i,j] == -1 :
                        print(colored("/",colors[self.grid[i,j]]), end=" ") # remplacement des -1 par des /
                    else :  
                        print(colored(self.grid[i,j],colors[self.grid[i,j]]), end=" ")
            print("\n")

    def init_grid(self) :
        pmur=0.15 
        pblanc=0.45 
        pverte=0.1
        pbleue=0.1
        prouge=0.1
        #  pnoire=0.1 mais pas besoin de le specifier c'est la couleur restante
        for i in range(self.nblignes):
            for j in range(self.nbcolonnes):
                z=np.random.uniform(0,1)
                if z < pmur:
                    c=-1
                else:
                    if z < pmur+ pblanc:
                        c=0
                    else:    
                        if z < pmur+ pblanc + pverte:
                            c=1
                        else:
                            if z < pmur+ pblanc +pverte + pbleue:
                                c=2
                            else:
                                if z< pmur + pblanc + pverte + pbleue +prouge:
                                    c=3
                                else:
                                    c=4   
                self.grid[i,j]=c
        self.grid[0,0]=0
        self.grid[0,1]=0
        self.grid[2,0]=0    
        self.grid[self.nblignes-1,self.nbcolonnes-1]=0 # Ajout du 0 derniere case en bas a droite
        # Ajout du 0 dans langle en bas a droite
        self.grid[self.nblignes-2,self.nbcolonnes-1]=0
        self.grid[self.nblignes-1,self.nbcolonnes-2]=0
        # Ajout de 0 pour avoir toujours un chemin accessible
        self.grid[self.nblignes-2,self.nbcolonnes-3]=0
        self.grid[self.nblignes-3,self.nbcolonnes-2]=0

    def reward(
        self,
        i : int,
        j : int
    ) :
        if self.isFinalcoords(i,j) :
            return 1e10
        else: 
            return - (1 + self.costs[self.grid[i,j]])        

    def replace_player_init(self):
        self.PosX,self.PosY = 0,0
        return 0

    def get_nearest_position(
        self,
        i : int, # index ligne
        j : int # index colonne
    ) :
        l = list()

        if i > 0 and j > 0 and self.grid[i-1,j-1] > -1 :
            state = (i-1,j-1)
            l.append(state)
        if i > 0 and self.grid[i-1,j] > -1 :
            state = (i-1,j)
            l.append(state)
        if i > 0 and j < self.nbcolonnes-1 and self.grid[i-1,j+1] > -1 :
            state = (i-1,j+1)
            l.append(state)
        if j > 0 and self.grid[i,j-1] > -1 :
            state = (i,j-1)
            l.append(state)
        if j < self.nbcolonnes-1 and self.grid[i,j+1] > -1 :
            state = (i,j+1)
            l.append(state)
        if i < self.nblignes-1 and j > 0 and self.grid[i+1,j-1] > -1 :
            state = (i+1,j-1)
            l.append(state)
        if i < self.nblignes-1 and self.grid[i+1,j] > -1 :
            state = (i+1,j)
            l.append(state)
        if i < self.nblignes-1 and j < self.nbcolonnes-1 and self.grid[i+1,j+1] > -1 :
            state = (i+1,j+1)
            l.append(state)

        return l

    def getMDP(self):

        """
        return : 
            {
                etat_1 : [ a_1, ... , a_n],
                ...
                etat_k : [ a_1, ... , a_n]
            }        
            ou a_i = [
                (new_state,proba,reward,done)_1
                ...
                (new_state,proba,reward,done)_m avec m <= 8
            ]
        """

        MDP = dict()
        for i in range(self.nblignes) :
            for j in range(self.nbcolonnes) :
                
                if self.grid[i,j] < 0 :
                    continue
            
                # print(self.grid[i,j])
                state = self.state_dict[(i,j)]
                list_action = list()
                for action in range(8):
                    list_states_recheable = list()

                    state_goal = self.make_action(action,i,j)
                    reward,new_li,new_cj,changed = state_goal
                    # print("reward : ",reward)
                    if changed :
                        new_state = self.state_dict[(new_li,new_cj)]
                        if not self.alea : 
                            list_states_recheable.append( (new_state , 1.0, reward , self.isFinalcoords(new_li,new_cj)) )
                        else :
                            # recupération dans une liste les états atteignable par aléatoire
                            l =  self.get_nearest_position(new_li,new_cj) 
                            q = len(l)
                            proba = 1.0 - (q/16)
                            list_states_recheable.append( (new_state , proba, reward , self.isFinalcoords(new_li,new_cj)) )

                            for (i_bis,j_bis) in l :
                                
                                new_state = self.state_dict[(i_bis,j_bis)]
                                proba = (q/16) / q
                                reward = self.reward(i_bis,j_bis)
                                list_states_recheable.append( (new_state , proba, reward , self.isFinalcoords(i_bis,j_bis)) )
                            
                        
                    else:
                        list_states_recheable.append( (self.state_dict[(i,j)] , 1.0, 0 , self.isFinalcoords(i,j)) )
                        
                    list_action.append(list_states_recheable)
                    
                MDP[state] = list_action
        return MDP



"""

env = EnvLabyrinthe(
    nblignes = 10,
    nbcolonnes = 20,
    alea = 0 # 0 : deterministe / 1 : stochastique
)

env.init_grid()

print("--------------------------------------------------")

MDP = env.getMDP()

print(f"Affichage des actions de la pos ({env.PosX,env.PosY}): ")
# print(MDP[(env.PosX,env.PosY)])


print("--------------------------------------------------")

env.render()
"""
