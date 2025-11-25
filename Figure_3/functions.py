import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy
import networkx as nx
import patsy
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import time
import os
import pickle

# put TAM_main into this directory
from TAM_main.utils import *



def edge2adj(edges, p):
    adj = np.zeros((p,p))
    for (i,j) in edges:
        adj[i,j] = 1
    return adj




def edge_diff(edges_star, edges_hat, p):
    return np.sum( np.abs( edge2adj(edges_star, p) - edge2adj(edges_hat, p) ) )




def gen_data(edges, n, p, func, sd = np.array([0.1])):
    if len(sd) == 1:
        sd = np.repeat(sd, p)
    
    adj_mat = edge2adj(edges, p)
    caus_order = compute_caus_order(adj_mat)
    X = np.zeros((n, p))
#     X[:, caus_order[0]] = np.random.normal(scale = 0.1, size = n)
    for i in range(p):
        j = caus_order[i]
        parent = np.where(adj_mat[:,j] == 1)[0]
        if(len(parent) == 0):
            X[:,j] = np.random.normal(scale = sd[j], size = n)
        else:
            Z = 0
            for pa in parent:
                # assign the function here
                Z = Z + func(X[:,pa])
            Z = Z - np.mean(Z, axis=0)
            X[:,j] = Z.reshape(-1) + np.random.normal(scale = sd[j], size = n)
    return(X)




def rbf_kernel_single(x1, x2, gamma=1.0):
    """
    Computes the RBF kernel between two individual data points.

    Args:
        x1 (np.ndarray): The first data point (1D array).
        x2 (np.ndarray): The second data point (1D array).
        gamma (float): The kernel parameter (default is 1.0).

    Returns:
        float: The RBF kernel similarity between x1 and x2.
    """
    squared_distance = np.linalg.norm(x1 - x2)**2
    return np.exp(-gamma * squared_distance)




def rbf_kernel(x, nb = 9, gamma = 1.0):
    n = x.size
    nb = 9
    space = 1 / (nb + 1)
    Q = space * np.arange(1, nb + 1)
    Z = np.zeros((n, len(Q)))
    for i in range(n):
        for q in Q:
            idx = np.where(q == Q)[0]
            Z[i,idx] = rbf_kernel_single(np.quantile(x, q), x[i], gamma = gamma)
    return Z




def gen_suff_stat(X, basis_setting):
    basis_type = basis_setting['type']
    
    if basis_type == 'spline':
        n_knots = basis_setting['n_knots']
        space = 1 / (n_knots + 1)
        Q = space * np.arange(1, n_knots + 1)
        degree = basis_setting['degree']
        include_intercept = basis_setting['include_intercept']
        def gen_basis(x):
            return patsy.bs(x, knots = np.quantile(x, Q), degree = degree, include_intercept = include_intercept)
    
    if basis_type == 'rbf_k':
        nb = basis_setting['nb']
        gamma = basis_setting['gamma']
        def gen_basis(x):
            return rbf_kernel(x, nb = nb, gamma = gamma)
    
    Xsp = {}
    n, p = X.shape
    for j in range(p):
        A = gen_basis(X[:,j])
        A = A - np.mean(A, axis = 0)
        Xsp[j] = A
    
    R = A.shape[1]
    
    Z = X
    for r in range(R):
        for j in range(p):
            Z = np.append(Z, Xsp[j][:,r].reshape(-1,1), axis = 1)
            
    Sigma = Z.T @ Z / n
    
    return {"Sigma": Sigma, "R": R}




def Mips_suff(Sigma_hat, n, p, lam, R, moral, time_limit = 300, known_edges = [], partial_order = [], start_edges = [], tau = 0.001, verbose = 1):
    
    # find the superset
    E = [(i, j) for i in range(p) for j in range(p) if i != j]  # off diagonal edge sets
    list_edges = []
    for edge in moral:
        list_edges.append((edge[0], edge[1]))
        list_edges.append((edge[1], edge[0]))
    G_moral = nx.Graph()
    for i in range(p):
        G_moral.add_node(i)
        G_moral.add_edges_from(list_edges)
    non_edges = list(set(E) - set(list_edges))
    
    # ! model setup !
    m = gp.Model()
    
    # ! Create variables !
    # Continuous variables
    # set Gamma
    Gamma = {}
    for i in range(p):
        for j in range(p):
            if i == j:
                Gamma[i,j] = m.addVar(lb=1e-5, vtype=GRB.CONTINUOUS, name="Gamma_%s_%s_%s" % (i, j, 0))
            else:
                Gamma[i,j] = 0
    for r in range(1,(R+1)):
        for j in range(p):
            for i in range(p):
                if (i,j) in list_edges:
                    Gamma[j,i+p*r] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Gamma_%s_%s_%s" % (i, j, r))
                else:
                    Gamma[j,i+p*r] = 0
    
    # set psi
    psi = m.addMVar((p, 1), lb=1, ub=p, vtype=GRB.CONTINUOUS, name='psi')
    
    # Integer variables
    g = {}
    for i,j in list_edges:
        if (i,j) in known_edges:
            g[i,j] = 1
        else:
            g[i,j] = m.addVar(vtype=GRB.BINARY, name="g_%s_%s" % (i, j))
        
    # Variables for outer approximation
    T = {}
    for i in range(p):
        T[i] = m.addVar(lb=-10, ub=100, vtype=GRB.CONTINUOUS, name="T_%s" % i)
        # This gives Gamma[i,i] a range about [0.0001, 100]
            
    for i in range(p):
        m.addGenConstrLog(Gamma[i,i], T[i])
    
    t1 = time.time()
    # ! Set objective !
    # log term
    log_term = gp.LinExpr()
    for i in range(p):
        log_term += -2 * T[i]
    
    # trace term
    trace = gp.QuadExpr()
    for k in range((R+1)*p):
        for i in range((R+1)*p):
            for j in range(p):
                trace += Gamma[j,k] * Gamma[j, i] * Sigma_hat[i, k]
    
    # penalty term
    penalty = gp.LinExpr()
    for i,j in list_edges:
        penalty += lam * g[i, j]
    
    t2 = time.time()
        
    # ! Solve the problem without constraints to get big_M !
    m.setObjective(log_term + trace + penalty, GRB.MINIMIZE)
    m.Params.OutputFlag = verbose
    m.Params.TimeLimit = time_limit
    m.update()
    t3 = time.time()
    m.optimize()
    t4 = time.time()
    
    big_M = 0
    for i in range(p):
        big_M = max(big_M, abs(Gamma[i, i].x))
    for r in range(1,(R+1)):
        for i,j in list_edges:
            big_M = max(big_M, abs(Gamma[j,i+p*r].x))
    M = 2*big_M
    
    # ! Set constraints !    
    m.addConstrs(Gamma[i, i] <= M for i in range(p))
    for i in range(p):
        m.addGenConstrLog(Gamma[i,i], T[i])
    
    for r in range(1,(R+1)):
        for i,j in list_edges:
            m.addConstr(M*g[i,j] >= Gamma[j,i+p*r])
            m.addConstr(-M*g[i,j] <= Gamma[j,i+p*r])
    
    for i,j in list_edges:
        m.addConstr(1-p+p*g[i, j] <= psi[j] - psi[i])
    
    for i,j in partial_order:
        m.addConstr(psi[i] <= psi[j])

        
    # ! solve the problem !
    m.setObjective(log_term + trace + penalty, GRB.MINIMIZE)
    m.Params.OutputFlag = verbose
    m.Params.TimeLimit = time_limit
    if tau > 0:
        m.Params.MIPGapAbs = tau
    m.update()
    t5 = time.time()
    m.optimize()
    t6 = time.time()

    # ! get vars !            
    g_val = []
    for var in m.getVars():
        if "g" in var.varName and var.X == 1:
            name = var.varName.split("_")[1:]
            g_val.append((int(name[0]), int(name[1])))
    for i,j in list_edges:
        if (i,j) in known_edges:
            g_val.append((i,j))
    
    
    return {"g":g_val, "time_obj": t2 - t1, "time_bigM": t4 - t3, "time_opt": t6 - t5}





def Mips_default(X, basis_mat, n, p, lam, R, moral, time_limit = 300, known_edges = [], partial_order = [], start_edges = [], tau = 0.001, verbose = 1):
    
    # find the superset
    E = [(i, j) for i in range(p) for j in range(p) if i != j]  # off diagonal edge sets
    list_edges = []
    for edge in moral:
        list_edges.append((edge[0], edge[1]))
        list_edges.append((edge[1], edge[0]))
    G_moral = nx.Graph()
    for i in range(p):
        G_moral.add_node(i)
        G_moral.add_edges_from(list_edges)
    non_edges = list(set(E) - set(list_edges))
    
    
    
    # ! model setup !
    m = gp.Model()
    
    # ! Create variables !
    # Continuous variables
    # set Gamma
    Gamma = {}
    for i in range(p):
        for j in range(p):
            for r in range(R):
                if (i,j) in list_edges:
                    Gamma[i,j,r] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Gamma_%s_%s_%s" % (i, j, 0))
                else:
                    Gamma[i,j,r] = 0
    
    # set psi
    psi = m.addMVar((p, 1), lb=1, ub=p, vtype=GRB.CONTINUOUS, name='psi')
    
    # set nu
    nu = m.addMVar((p, 1), lb=1e-5, vtype=GRB.CONTINUOUS, name='nu')
    
    # Integer variables
    g = {}
    for i,j in list_edges:
        if (i,j) in known_edges:
            g[i,j] = 1
        else:
            g[i,j] = m.addVar(vtype=GRB.BINARY, name="g_%s_%s" % (i, j))
    
    
    # Variables for outer approximation
    T = {}
    for i in range(p):
        T[i] = m.addVar(lb=-10, ub=100, vtype=GRB.CONTINUOUS, name="T_%s" % i)
        # This gives Gamma[i,i] a range about [0.0001, 100]
            
    for i in range(p):
        m.addGenConstrLog(nu[i], T[i])
    
    
    t1 = time.time()
    # ! Set objective !
    # log term
    log_term = gp.LinExpr()
    for i in range(p):
        log_term += -2 * T[i]
    
    # quad term
    quad = gp.QuadExpr()
    for j in range(p):
        for i in range(n):
            termji = nu[j] * X[i,j]
            for k in range(p):
                for r in range(R):
                    termji -= basis_mat[k][i,r] * Gamma[k,j,r]
            quad += termji ** 2
    quad = quad / n
    
    
    # penalty term
    penalty = gp.LinExpr()
    for i,j in list_edges:
        penalty += lam * g[i, j]
    
    
    t2 = time.time()
    
        
    # ! Solve the problem without constraints to get big_M !
    m.setObjective(log_term + quad + penalty, GRB.MINIMIZE)
    m.Params.OutputFlag = verbose
    m.Params.TimeLimit = time_limit
    m.update()
    t3 = time.time()
    m.optimize()
    t4 = time.time()
    
    
    big_M = 0
    for r in range(R):
        for i,j in list_edges:
            big_M = max(big_M, abs(Gamma[i,j,r].x))
            
    for j in range(p):
        big_M = max(big_M, nu[j].x)
        
    M = 2*big_M
    
    
    
    # ! Set constraints !    
    for r in range(R):
        for i,j in list_edges:
            m.addConstr(M*g[i,j] >= Gamma[i,j,r])
            m.addConstr(-M*g[i,j] <= Gamma[i,j,r])
    
    for j in range(p):
        nu[j] <= M
    
    for i,j in list_edges:
        m.addConstr(1-p+p*g[i, j] <= psi[j] - psi[i])
    
    for i,j in partial_order:
        m.addConstr(psi[i] <= psi[j])
    
    
        
    # ! solve the problem !
    m.setObjective(log_term + quad + penalty, GRB.MINIMIZE)
    m.Params.OutputFlag = verbose
    m.Params.TimeLimit = time_limit
    m.Params.DualReductions = 0
    m.Params.MIPFocus = 1
    if tau > 0:
        m.Params.MIPGapAbs = tau
    m.update()
    t5 = time.time()
    m.optimize()
    t6 = time.time()
    
    g_val = []
    for var in m.getVars():
        if "g" in var.varName and var.X == 1:
            name = var.varName.split("_")[1:]
            g_val.append((int(name[0]), int(name[1])))
    for i,j in list_edges:
        if (i,j) in known_edges:
            g_val.append((i,j))
    
    return {"g":g_val, "time_obj": t2 - t1, "time_bigM": t4 - t3, "time_opt": t6 - t5}





def gen_basis_matrix(X, basis_setting):
    basis_type = basis_setting['type']
    
    if basis_type == 'spline':
        n_knots = basis_setting['n_knots']
        space = 1 / (n_knots + 1)
        Q = space * np.arange(1, n_knots + 1)
        degree = basis_setting['degree']
        include_intercept = basis_setting['include_intercept']
        def gen_basis(x):
            return patsy.bs(x, knots = np.quantile(x, Q), degree = degree, include_intercept = include_intercept)
    
    if basis_type == 'rbf_k':
        nb = basis_setting['nb']
        gamma = basis_setting['gamma']
        def gen_basis(x):
            return rbf_kernel(x, nb = nb, gamma = gamma)
    
    
    basis_mat = {}
    n, p = X.shape
    for j in range(p):
        A = gen_basis(X[:,j])
        A = A - np.mean(A, axis = 0)
        basis_mat[j] = A
    
    R = A.shape[1]
    
    return {"basis_mat": basis_mat, "R": R}







# dataset!!!!!
# small
dsep = {'name': 'dsep', 'p': 6, 
        'edges': [(1,3), (1,5), (2,3), (2,4), (3,5), (5,6)],
        'moral': [(1,2), (1,3), (1,5), (2,3), (2,4), (3,5), (5,6)]}

asia = {'name': 'asia', 'p': 8, 
        'edges': [(1,2), (2,5), (3,4), (3,6), (4,5), (5,7), (5,8), (6,7)], 
        'moral': [(1,2), (2,4), (2,5), (3,4), (3,6), (4,5), (5,6), (5,7), (5,8), (6,7)]}

bowling = {'name': 'bowling', 'p': 9, 
           'edges': [(1,3), (2,3), (3,6), (4,6), (4,7), (5,7), (5,8), (6,8), (6,9), (7,9), (8,9)],
           'moral': [(1,2), (1,3), (2,3), (3,4), (3,6), (4,5), (4,6), (4,7), (5,6), (5,7), (5,8), (6,7), (6,8), 
                     (6,9), (7,8), (7,9), (8,9)]}

inssmall = {'name': 'inssmall', 'p': 15, 
            'edges':[(1,2), (1,3), (2,3), (2,4), (2,13), (2,14), (3,4), (3,5), (3,9), (4,5), (4,9), (4,10), (4,14), 
                     (5,7), (5,12), (6,7), (6,8), (6,15), (7,9), (8,10), (8,11), (8,12), (9,12), (10,13), (13,14)],
            'moral':[(1,2), (1,3), (2,3), (2,4), (2,10), (2,13), (2,14), (3,4), (3,5), (3,7), (3,9), (4,5), (4,7), 
                     (4,8), (4,9), (4,10), (4,13), (4,14), (5,6), (5,7), (5,8), (5,9), (5,12), (6,7), (6,8), (6,15), 
                     (7,9), (8,9), (8,10), (8,11), (8,12), (9,12), (10,13), (13,14)]}

rain = {'name': 'rain', 'p': 14, 
        'edges': [(1,2), (1,3), (1,4), (2,9), (4,5), (4,11), (4,14), (5,6), (5,7), (5,8), (6,11), (7,14), (8,14), 
                  (9,10), (10,14), (11,12), (11,13), (11,14)],
        'moral': [(1,2), (1,3), (1,4), (2,9), (4,5), (4,6), (4,7), (4,8), (4,10), (4,11), (4,14), (5,6), (5,7), 
                  (5,8), (6,11), (7,8), (7,10), (7,11), (7,14), (8,10), (8,11), (8,14), (9,10), (10,11), (10,14), 
                  (11,12), (11,13), (11,14)]}

cloud = {'name': 'cloud', 'p': 16, 
         'edges': [(1,2), (1,3), (2,5), (3,4), (3,6), (5,6), (5,13), (5,14), (6,7), (7,8), (7,9), (7,12), (7,15), 
                   (8,9), (8,10), (9,10), (10,11), (11,12), (15,16)],
         'moral': [(1,2), (1,3), (2,5), (3,4), (3,5), (3,6), (5,6), (5,13), (5,14), (6,7), (7,8), (7,9), (7,11), 
                   (7,12), (7,15), (8,9), (8,10), (9,10), (10,11), (11,12), (15,16)]}

funnel = {'name': 'funnel', 'p': 18, 
          'edges': [(1,2), (2,4), (3,4), (4,7), (4,9), (4,10), (5,7), (5,9), (6,7), (6,14), (8,9), (9,13), (11,12), 
                    (12,18), (13,18), (15,18), (16,17), (17,18)],
          'moral': [(1,2), (2,3), (2,4), (3,4), (4,5), (4,6), (4,7), (4,8), (4,9), (4,10), (5,6), (5,7), (5,8), 
                    (5,9), (6,7), (6,14), (8,9), (9,13), (11,12), (12,13), (12,15), (12,17), (12,18), (13,15), 
                    (13,17), (13,18), (15,17), (15,18), (16,17), (17,18)]}

galaxy = {'name': 'galaxy', 'p': 20, 
          'edges': [(1,4), (2,4), (3,4), (4,5), (5,6), (5,15), (6,7), (6,10), (6,12), (7,10), (7,11), (8,9), (8,10), 
                    (10,20), (11,18), (12,13), (12,14), (12,15), (14,16), (15,16), (17,18), (18,19)],
          'moral': [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4), (4,5), (5,6), (5,12), (5,15), (6,7), (6,8), (6,10), 
                    (6,12), (7,8), (7,10), (7,11), (8,9), (8,10), (10,20), (11,17), (11,18), (12,13), (12,14), 
                    (12,15), (14,15), (14,16), (15,16), (17,18), (18,19)]}


# medium
insurance = {'name': 'insurance', 'p': 27, 
             'edges': [(1,2), (1,3), (2,3), (2,4), (2,13), (2,14), (2,23), (3,4), (3,5), (3,9), (3,18), (3,19), 
                       (3,22), (4,5), (4,9), (4,10), (4,14), (4,18), (4,19), (4,27), (5,7), (5,12), (5,17), (5,25), 
                       (6,7), (6,8), (6,15), (7,9), (7,21), (7,24), (8,10), (8,11), (8,12), (8,21), (8,23), (8,26),
                       (9,12), (9,17), (9,25), (10,13), (11,17), (13,14), (13,27), (15,16), (15,17), (15,20), 
                       (16,17), (16,18), (16,19), (20,21), (23,24), (24,25)], 
             'moral': [(1,2), (1,3), (2,3), (2,4), (2,8), (2,10), (2,13), (2,14), (2,23), (3,4), (3,5), (3,7), 
                       (3,9), (3,16), (3,18), (3,19), (3,22), (4,5), (4,7), (4,8), (4,9), (4,10), (4,13), (4,14), 
                       (4,16), (4,18), (4,19), (4,27), (5,6), (5,7), (5,8), (5,9), (5,11), (5,12), (5,15), (5,16), 
                       (5,17), (5,24), (5,25), (6,7), (6,8), (6,15), (7,8), (7,9), (7,20), (7,21), (7,23), (7,24), 
                       (8,9), (8,10), (8,11), (8,12), (8,20), (8,21), (8,23), (8,26), (9,11), (9,12), (9,15), 
                       (9,16), (9,17), (9,24), (9,25), (10,13), (11,15), (11,16), (11,17), (13,14), (13,27), 
                       (15,16), (15,17), (15,20), (16,17), (16,18), (16,19), (20,21), (23,24), (24,25)]}

factors = {'name': 'factors', 'p': 27, 
           'edges': [(1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), (1,10), (1,11), (1,12), (1,13), (1,14), 
                     (1,15), (1,16), (1,17), (1,18), (1,19), (1,20), (1,21), (1,22), (1,23), (1,24), (1,25), (1,26), 
                     (1,27), (2,4), (2,6), (2,8), (2,10), (2,12), (2,14), (2,16), (2,18), (2,20), (2,22), (2,24), 
                     (2,26), (3,6), (3,9), (3,12), (3,15), (3,18), (3,21), (3,24), (3,27), (4,8), (4,12), (4,16), 
                     (4,20), (4,24), (5,10), (5,15), (5,20), (5,25), (6,12), (6,18), (6,24), (7,14), (7,21), (8,16), 
                     (8,24), (9,18), (9,27), (10,20), (11,22), (12,24), (13,26)],
           'moral': [(1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), (1,10), (1,11), (1,12), (1,13), (1,14), 
                     (1,15), (1,16), (1,17), (1,18), (1,19), (1,20), (1,21), (1,22), (1,23), (1,24), (1,25), (1,26), 
                     (1,27), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (2,9), (2,10), (2,11), (2,12), (2,13), 
                     (2,14), (2,16), (2,18), (2,20), (2,22), (2,24), (2,26), (3,4), (3,5), (3,6), (3,7), (3,8), 
                     (3,9), (3,12), (3,15), (3,18), (3,21), (3,24), (3,27), (4,5), (4,6), (4,8), (4,10), (4,12), 
                     (4,16), (4,20), (4,24), (5,10), (5,15), (5,20), (5,25), (6,8), (6,9), (6,12), (6,18), (6,24), 
                     (7,14), (7,21), (8,12), (8,16), (8,24), (9,18), (9,27), (10,20), (11,22), (12,24), (13,26)]}


