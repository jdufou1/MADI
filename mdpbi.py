#!/usr/bin/python

# Copyright 2013, Gurobi Optimization, Inc.

from gurobipy import *

nbcont=2 
nbvar=4

# Range of plants and warehouses
lignes = range(nbcont)
colonnes = range(nbvar)


# Matrice des contraintes
a = [[5.0/8.0,-1.0/8.0],
[3.0/4.0, -1.0/4.0],
[-1.0/4.0, 3.0/4.0],
[-1.0/8.0, 5.0/8.0]]


# Second membre
b = [1.0, 0.0]

# Coefficients de la fonction objectif
c = [[8.0, 12.0, 11.0,9.0],
 [13.0, 6.0, 7.0,15.0]]

m = Model("mdp")     
        
# declaration variables de decision
x = []
for i in colonnes:
    x.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="x%d" %(i+1)))
z=m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="z")

# maj du modele pour integrer les nouvelles variables
m.update()

obj = LinExpr()
obj = z
    
# definition de l'objectif
m.setObjective(obj,GRB.MAXIMIZE)

l=[1.0,1.0]

# Definition des contraintes
for i in lignes:
    m.addConstr(quicksum(a[j][i]*x[j] for j in colonnes) == b[i], "Contrainte %d" % i)
    m.addConstr(quicksum(c[i][j]*x[j]*l[i] for j in colonnes) - z >= 0, "Controbj %d" % i)
# Resolution
m.optimize()
print("")                
print('Valeurs de la politique optimale:\n')
print('x11 = ',x[0].x)
print('x12 = ', x[1].x)
print('x21 = ',x[2].x)
print('x22 = ', x[3].x)
print('Valeur de la fonction objectif :', m.objVal)
print("")
sum1=x[0].x + x[1].x
sum2=x[2].x + x[3].x
print('Valeurs des probabilites :')
print('p11 = ',x[0].x/sum1)
print('p12 = ', x[1].x/sum1)
print('p21 = ',x[2].x/sum2)
print('p22 = ', x[3].x/sum2)

   