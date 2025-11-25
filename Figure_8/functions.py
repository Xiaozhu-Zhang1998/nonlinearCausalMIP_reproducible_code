import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy
import networkx as nx
import patsy
import random
from celer import GroupLassoCV, GroupLasso
from sklearn.preprocessing import StandardScaler
from cdt.causality.graph import CAM
from cdt.causality.graph import CCDr
import micodag as mic
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import time
import os
import pickle


# put the folder `TAM_main` into this directory
from TAM_main.utils import *

# put the folder `notears` into this directory
from notears.nonlinear import *
import notears.utils as ut





# helper functions
def edge2adj(edges, p):
    adj = np.zeros((p,p))
    for (i,j) in edges:
        adj[i,j] = 1
    return adj





def edge_diff(edges_star, edges_hat, p):
    return np.sum( np.abs( edge2adj(edges_star, p) - edge2adj(edges_hat, p) ) )





def isCyclicUtil(adj, u, visited, recStack):
    # If the node is already in the current recursion stack, a cycle is detected
    if recStack[u]:
        return True

    # If the node is already visited and not part of the recursion stack, skip it
    if visited[u]:
        return False

    # Mark the current node as visited and add it to the recursion stack
    visited[u] = True
    recStack[u] = True

    # Recur for all the adjacent vertices
    for v in adj[u]:
        if isCyclicUtil(adj, v, visited, recStack):
            return True

    # Remove the node from the recursion stack before returning
    recStack[u] = False
    return False





# Function to build adjacency list from edge list
def constructadj(V, edges):
    adj = [[] for _ in range(V)]  # Create a list for each vertex
    for u, v in edges:
        adj[u].append(v)  # Add directed edge from u to v
    return adj





# Main function to detect cycle in the directed graph
def isCyclic(V, edges):
    adj = constructadj(V, edges)
    visited = [False] * V       # To track visited vertices
    recStack = [False] * V      # To track vertices in the current DFS path

    # Try DFS from each vertex
    for i in range(V):
        if not visited[i] and isCyclicUtil(adj, i, visited, recStack):
            return True  # Cycle found
    return False  # No cycle found





# how to generate data systematically?
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





## by using the splines
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





def groupLasso(Z, y, groups, alphas = np.array([0.01, 0.005, 0.004, 0.003, 0.002, 0.001]), alpha_id = None):
    if alpha_id == None:
        # GroupLassoCV
        gl_cv = GroupLassoCV(groups = groups, cv = 5, alphas = np.sort(alphas)[::-1], fit_intercept = False)
        gl_cv.fit(Z, y)
        
        # find the 1se.cv alpha
        alphas = gl_cv.alphas_
        mse_path = gl_cv.mse_path_
        if mse_path.ndim == 2:
            mse_mean = mse_path.mean(axis=1)
            mse_se = mse_path.std(axis=1, ddof=1) / np.sqrt(mse_path.shape[1])
            i_min = np.argmin(mse_mean)
            mse_min, se_min = mse_mean[i_min], mse_se[i_min]
            candidates = np.where(mse_mean <= mse_min + se_min)[0]
            i_1se = candidates.min()
            alpha_id = alphas[i_1se]
        else:
            alpha_id = alphas[0]
        
        # Re-fit GroupLasso
        gl1se = GroupLasso(groups=groups, alpha=alpha_id, fit_intercept = False,
                           max_iter = gl_cv.max_iter, tol = gl_cv.tol, verbose = gl_cv.verbose)
    
    else:
        gl1se = GroupLasso(groups = groups, alpha=alpha_id, fit_intercept = False)
    
    gl1se.fit(Z, y)
    active_group = np.unique(np.nonzero(gl1se.coef_)[0] // groups)

    return {'active_group': active_group, 'alpha_id':alpha_id}





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





def find_moral(W, basis_setting, alphas = np.array([0.01, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0001])):
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
        
    moral = set({})
    n, p = W.shape
    
    for i in range(p):
        # find X and y
        y = W[:,i]
        X = np.delete(W, i, axis=1)
        
        # find basis functions
        p_ = X.shape[1]
        Z = np.zeros((n, 0))
        for j in range(p_):
            A = gen_basis(X[:,j])
            A = A - np.mean(A, axis = 0)
            Z = np.append(Z, A, axis = 1)
        groups = A.shape[1]
        
        # find selections set
        selset = groupLasso(Z, y, groups, alphas)
        feat_id = np.setdiff1d(range(p), i)
        for k in feat_id[selset['active_group']]:
            x = np.sort((i,k))
            moral.add(tuple(x))
            
    return(moral)






def loss_from_edges(edges, W_model, W_val, basis_setting):
    n, p = W_val.shape
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
    
    # find adjacency matrix
    Adj_mat = edge2adj(edges, p)
    var_hat = np.array([])
    
    # find variances
    for i in range(p):
        idx = np.nonzero(Adj_mat[:,i])[0]
        if(idx.size != 0):
            y_model = W_model[:,i]
            X_model = W_model[:,idx]
            y_val = W_val[:,i]
            X_val = W_val[:,idx]
            # find basis functions
            p_ = X_model.shape[1]
            Z_model = np.zeros((n, 0))
            Z_val = np.zeros((n, 0))
            for j in range(p_):
                A_model = gen_basis(X_model[:,j])
                A_model = A_model - np.mean(A_model, axis = 0)
                A_val = gen_basis(X_val[:,j])
                A_val = A_val - np.mean(A_val, axis = 0)
                Z_model = np.append(Z_model, A_model, axis = 1)
                Z_val = np.append(Z_val, A_val, axis = 1)
            r = y_val - Z_val @ np.linalg.inv(Z_model.T @ Z_model) @ Z_model.T @ y_model
        else:
            r = W_val[:,i]
        
        var_hat = np.append(var_hat, np.square(np.linalg.norm(r, ord = 2)) / n )
    
    # find loss value
    loss_val = np.sum(np.log(var_hat)) + p
    return(loss_val)





# write a function for our method
def Mips_DAG_equalvar(Sigma_hat, n, p, lam, R, moral, time_limit = 300, known_edges = [], partial_order = [], start_edges = [], tau = 0.001, verbose = 1):
    
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
    # set Gamma (which is essentially Beta)
    Gamma = {}
    for i in range(p):
        for j in range(p):
            if i == j:
                Gamma[i,j] = 1
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
            
    # ! Set objective !
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
        
    # ! Solve the problem without constraints to get big_M !
    m.setObjective(trace + penalty, GRB.MINIMIZE)
    m.Params.OutputFlag = verbose
    m.Params.TimeLimit = time_limit
    m.update()
    m.optimize()
    
    big_M = 0
    for r in range(1,(R+1)):
        for i,j in list_edges:
            big_M = max(big_M, abs(Gamma[j,i+p*r].x))
    M = 2*big_M
    
    # ! Set constraints !    
    for r in range(1,(R+1)):
        for i,j in list_edges:
            m.addConstr(M*g[i,j] >= Gamma[j,i+p*r])
            m.addConstr(-M*g[i,j] <= Gamma[j,i+p*r])
    
    for i,j in list_edges:
        m.addConstr(1-p+p*g[i, j] <= psi[j] - psi[i])
    
    for i,j in partial_order:
        m.addConstr(psi[i] <= psi[j])
    
        
    # ! solve the problem !
    m.setObjective(trace + penalty, GRB.MINIMIZE)
    # m.Params.TimeLimit = timelimit*p
    m.Params.OutputFlag = verbose
    m.Params.TimeLimit = time_limit
    if tau > 0:
        m.Params.MIPGapAbs = tau
    m.update()
    m.optimize()
    
    # ! get vars !
    Beta_val = np.zeros((p, p * (R+1)))
    for var in m.getVars():
        if "Gamma" in var.varName:
            name = var.varName.split("_")[1:]
            i = int(name[0])
            j = int(name[1])
            r = int(name[2])
            Beta_val[j,i+p*r] = var.X
    for i in range(p):
        Beta_val[i,i] = 1
    
    g_val = []
    for var in m.getVars():
        if "g" in var.varName and var.X == 1:
            name = var.varName.split("_")[1:]
            g_val.append((int(name[0]), int(name[1])))
    for i,j in list_edges:
        if (i,j) in known_edges:
            g_val.append((i,j))
            
    psi_val = {}
    for var in m.getVars():
        if "psi" in var.varName:
            name = var.varName.split('[')[1].split(']')[0].split(',')
            psi_val[int(name[0])] = var.X
            
    # how to estimate variance?
    varhat = np.sum(np.diag(Beta_val @ Sigma_hat @ Beta_val.T)) / p
    Gamma_val = Beta_val / np.sqrt(varhat)

    # how to get the loss function?
    loss_val = p * np.log(varhat) + p
    
    return {"Beta": Beta_val, "Gamma": Gamma_val, "g":g_val, "psi":psi_val, "var": varhat, "loss_val": loss_val}







# write a function for our method
def Mips_DAG_diffvar(Sigma_hat, n, p, lam, R, moral, time_limit = 300, known_edges = [], partial_order = [], start_edges = [], tau = 0.001, verbose = 1):
    
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
        
    # ! Solve the problem without constraints to get big_M !
    m.setObjective(log_term + trace + penalty, GRB.MINIMIZE)
    m.Params.OutputFlag = verbose
    m.Params.TimeLimit = time_limit
    m.update()
    m.optimize()
    
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
    # m.Params.TimeLimit = timelimit*p
    m.Params.OutputFlag = verbose
    m.Params.TimeLimit = time_limit
    if tau > 0:
        m.Params.MIPGapAbs = tau
    m.update()
    m.optimize()

    # ! get vars !
    Gamma_val = np.zeros((p, p * (R+1)))
    for var in m.getVars():
        if "Gamma" in var.varName:
            name = var.varName.split("_")[1:]
            i = int(name[0])
            j = int(name[1])
            r = int(name[2])
            Gamma_val[j,i+p*r] = var.X
            
    g_val = []
    for var in m.getVars():
        if "g" in var.varName and var.X == 1:
            name = var.varName.split("_")[1:]
            g_val.append((int(name[0]), int(name[1])))
    for i,j in list_edges:
        if (i,j) in known_edges:
            g_val.append((i,j))
            
    psi_val = {}
    for var in m.getVars():
        if "psi" in var.varName:
            name = var.varName.split('[')[1].split(']')[0].split(',')
            psi_val[int(name[0])] = var.X
    
    # how to estimate variance?
    varhat = 1 / ( np.diag(Gamma_val) ** 2 )
    Beta_val = Gamma_val * np.sqrt(varhat)[:, np.newaxis]
    
    # how to get the loss function?
    loss_val = np.sum(np.log(varhat)) + p
    
    return {"Beta": Beta_val, "Gamma": Gamma_val, "g":g_val, "psi":psi_val, "varhat": varhat, "loss_val": loss_val}





def CAM_cv(W_train, edges_star, basis_setting_eval, 
           Cutoff = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]):
    
    n, p = W_train.shape
    
    Loss_Val = np.array([])
    for cutoff in Cutoff:
        camobj = CAM(score='nonlinear', cutoff=cutoff, variablesel=True, pruning=True)
        output = camobj.predict(pd.DataFrame(W_train))
        loss_val = loss_from_edges(list(output.edges), W_train, W_train, basis_setting_eval) + len(list(output.edges)) * np.log(n) / n
        Loss_Val = np.append(Loss_Val, loss_val)
        edge_diff_obj = edge_diff(edges_star,list(output.edges), p)
        print("CAM_cv: cutoff", cutoff, output, loss_val, edge_diff_obj)
        
    
    idx = np.argmin(Loss_Val)
    cutoff = Cutoff[idx]
    t1 = time.time()
    camobj = CAM(score='nonlinear', cutoff=cutoff, variablesel=True, pruning=True)
    output = camobj.predict(pd.DataFrame(W_train))
    t2 = time.time()
    
    return {'edges':list(output.edges), 'time': t2 - t1, 'cutoff': cutoff, 'loss_val': Loss_Val[idx]}






def MIP_pre_process(W_train, data_name, m, equalvar, basis_setting,
                    moral = None, alphas = None, s0_max = None,                   # for moral
                    cutoff = None, Loss_CAM = None, B = 20, kappa_par = 0.85, kappa_stab = 0.9,
                    reverse = False):   # for bootstrapping
    
    _,p = W_train.shape
    
    
    # bootstrapping using NPVAR
    if equalvar:
        name = './dataset/' + data_name['name'] + '_equalvar/Loss_NPVAR_' + str(m) + '.txt'
    else:
        name = './dataset/' + data_name['name'] + '_diffvar/Loss_NPVAR_' + str(m) + '.txt'
    with open(name, 'r') as file:
        Loss_NPVAR = float(file.read())
    
    print("Loss_CAM: ", Loss_CAM)
    print("Loss_NPVAR: ", Loss_NPVAR)

    condition = Loss_CAM > Loss_NPVAR
    
    if condition != reverse:
        print("used NPVAR", "\n")
        loss_val = Loss_NPVAR
        if equalvar:
            name = './dataset/' + data_name['name'] + '_equalvar/prob_adj_NPVAR_' + str(m) + '.csv'
        else:
            name = './dataset/' + data_name['name'] + '_diffvar/prob_adj_NPVAR_' + str(m) + '.csv'
        prob_adj = np.loadtxt(name, delimiter=',')
        CAM_bootstrap_time = None
    
    else:
        print("used CAM", "\n")
        loss_val = Loss_CAM
        if equalvar:
            name = './dataset/' + data_name['name'] + '_equalvar/prob_adj_CAM_' + str(m) + '.csv'
            name_time = './dataset/' + data_name['name'] + '_equalvar/bootstrap_time_CAM' + str(m) + '.csv'
        else:
            name = './dataset/' + data_name['name'] + '_diffvar/prob_adj_CAM_' + str(m) + '.csv'
            name_time = './dataset/' + data_name['name'] + '_diffvar/bootstrap_time_CAM' + str(m) + '.csv'
        if os.path.exists(name):
            prob_adj = np.loadtxt(name, delimiter=',')
            with open(name_time, 'r') as file:
                CAM_bootstrap_time = float(file.read())
        else:
            # bootstrapping using CAM
            print("started bootstrapping CAM...")
            t1_bootstrap = time.time()
            Adj = np.zeros((p,p))
            for b in range(B):
                idx = np.random.choice(list(range(W_train.shape[0])), W_train.shape[0], replace = True)
                W_idx = W_train[idx,:]
                camobj = CAM(score='nonlinear', cutoff=cutoff, variablesel=True, pruning=True)
                output = camobj.predict(pd.DataFrame(W_idx))
                Adj = Adj + edge2adj(list(output.edges), p)
                print("finished", b)
            prob_adj = Adj / B
            t2_bootstrap = time.time()
            print("finished bootstrapping CAM...", "\n")
            np.savetxt(name, prob_adj, delimiter=',')
            CAM_bootstrap_time = t2_bootstrap - t1_bootstrap
            with open(name_time, 'w') as file:
                file.write(str(CAM_bootstrap_time))


    
    # find superset
    if moral is None:
        # extract from bootstrapping
        for kappa_moral in np.linspace(0.1, 0.5, 15):
            super_edges = [tuple(np.sort((i,j))) for i in range(p) for j in range(p) if prob_adj[i,j] >= kappa_moral]
            super_edges = set(super_edges)
            if len(super_edges) <= s0_max:
                break
        superset_boot = super_edges

        # neighborhood selection
        while True:
            superset_NS = find_moral(W_train, basis_setting, alphas = alphas)
            if len(superset_NS) <= s0_max or alphas.size == 1:
                break
            else:
                alphas = alphas[:-1]
                print(len(superset_NS))

        moral = list(superset_boot.union(set(superset_NS)))
        print("found moral set")


    # find partial order set
    par_edges = list()
    par_vals = np.array([])
    for i,j in moral:
        if prob_adj[i,j] >= kappa_par and prob_adj[j,i] <= 1 - kappa_par:
            par_edges.append((i,j))
            par_vals = np.append(par_vals, prob_adj[i,j])
        if prob_adj[j,i] >= kappa_par and prob_adj[i,j] <= 1 - kappa_par:
            par_edges.append((j,i))
            par_vals = np.append(par_vals, prob_adj[j,i])
    sorted_idx = np.argsort(-par_vals)
    par_vals = par_vals[sorted_idx]
    par_edges = [par_edges[i] for i in sorted_idx]
    i = 1
    while i < len(par_vals):
        if isCyclic(p, par_edges[0:i]):
            par_vals = np.delete(par_vals, i)
            par_edges.pop(i)
        else:
            i += 1
    print("found partial order set")
    
    # find stable set
    stab_edges = list()
    stab_vals = np.array([])
    for i in range(p):
        for j in range(p):
            if prob_adj[i,j] >= kappa_stab and (i,j) in par_edges:
                stab_edges.append((i,j))
                stab_vals = np.append(stab_vals, prob_adj[i,j])
    sorted_idx = np.argsort(-stab_vals)
    stab_vals = stab_vals[sorted_idx]
    stab_edges = [stab_edges[i] for i in sorted_idx]
    i = 1
    while i < len(stab_vals):
        if isCyclic(p, stab_edges[0:i]):
            stab_vals = np.delete(stab_vals, i)
            stab_edges.pop(i)
        else:
            i += 1
    print("found stable set")
            
    return {'moral': moral, 'par_edges': par_edges, 'stab_edges': stab_edges, 'prob_adj': prob_adj, 'CAM_bootstrap_time': CAM_bootstrap_time,
            'loss_val': loss_val}





def MIP_cv(W_train, edges_star, basis_setting,
           moral = [], stab_edges = [], par_edges = [], 
           Lam = [0.01, 0.005, 0.001, 0.0005, 0.0001], equalvar = True, tau_const = 1, time_limit = 300):
    
    n, p = W_train.shape
    s0_max = len(moral)
    
    if equalvar:
        MIP_fun = Mips_DAG_equalvar
    else:
        MIP_fun = Mips_DAG_diffvar
    
    # find suff stat
    suff = gen_suff_stat(W_train, basis_setting)
    Sigma_hat = suff["Sigma"]
    R = suff["R"]
    
    # go!
    Loss_Val = np.array([])
    start_edges = []
    for lam in Lam:
        t3 = time.time()
        Mip_obj = MIP_fun(Sigma_hat, n, p, lam, R, moral, known_edges = stab_edges, partial_order = par_edges, start_edges = start_edges,
                          tau = lam * s0_max * tau_const, verbose = 0, time_limit = time_limit)
        t4 = time.time()
        start_edges = Mip_obj['g']
        loss_val = Mip_obj['loss_val'] + len(Mip_obj['g']) * np.log(n) / n
        edge_diff_obj = edge_diff(edges_star, Mip_obj['g'], p)
        Loss_Val = np.append(Loss_Val, loss_val)
        print("MIP_cv: lam", lam, loss_val, edge_diff_obj, t4 - t3, Mip_obj['g'])
    
    idx = np.argmin(Loss_Val)
    lam = Lam[idx]
    t3 = time.time()
    Mip_obj = MIP_fun(Sigma_hat, n, p, lam, R, moral, known_edges = stab_edges, partial_order = par_edges, 
                      tau = lam * s0_max * tau_const, verbose = 0, time_limit = time_limit)
    t4 = time.time()
    print("The selected edges:", Mip_obj['g'])
    
    return {'edges': Mip_obj['g'], 'time': t4 - t3, 'moral_size': len(moral), 'layer': Mip_obj['psi'], 'loss_val': Loss_Val[idx] }




def notears_cv(W_train, edges_star, basis_setting_eval, Lam = np.array([0.02, 0.01, 0.008, 0.005]) ):
    n,p = W_train.shape
    Loss_Val = np.array([])
    EDGES = []
    TIME = []
    
    for lam in Lam:
        t1 = time.time()
        torch.set_default_dtype(torch.double)
        np.set_printoptions(precision=3)
        model = NotearsMLP(dims=[p, 5, 1], bias=True)
        W_est = notears_nonlinear(model, W_train, lambda1=lam, lambda2=lam)
        t2 = time.time()
        est_edges = [(i,j) for i in range(p) for j in range(p) if W_est[i,j] !=0 ]
        EDGES.append(est_edges)
        TIME.append(t2 - t1)

        loss_val = loss_from_edges(est_edges, W_train, W_train, basis_setting_eval) + len(est_edges) * np.log(n) / n
        edge_diff_obj = edge_diff(edges_star, est_edges, p)
        Loss_Val = np.append(Loss_Val, loss_val)
        print("notears_cv: lam", lam, loss_val, edge_diff_obj, t2 - t1)
    idx = np.nanargmin(Loss_Val)
    lam = Lam[idx]
    est_edges = EDGES[idx]

    return {'edges': est_edges, 'time': TIME[idx], 'loss_val': Loss_Val[idx] }







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


causal_chamber = {'name':'causal_chamber', 'p': 8, 
                  'edges': [(1,4), (1,5), (1,6), (1,7), (2,4), (2,5), (2,6), (2,7), (3,4), (3,5), (3,6), (3,7), (8,7)]}