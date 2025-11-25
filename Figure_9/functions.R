loss_from_adj = function(Adj, W_model, W_val) {
  n = nrow(W_val)
  p = ncol(W_val)
  var_hat = c()
  
  for(i in 1:p){
    idx = which(Adj[,i] !=0)
    if(length(idx) != 0) {
      y_model = W_model[,i]
      X_model = W_model[,idx]
      y_val = W_val[,i]
      X_val = W_val[,idx]
      r = y_val - X_val %*% solve(t(X_model) %*% X_model) %*% t(X_model) %*% y_model
    } else {
      r = W_val[, i]
    }
    var_hat = c(var_hat, norm(r, "2")^2 / n )
  }
  
  # find loss value
  loss_val = sum(log(var_hat)) + p
  
  return(loss_val)
}




edge2adj = function(edges, p) {
  adj = matrix(0, p, p)
  for(i in 1:length(edges)) {
    adj[edges[[i]][1], edges[[i]][2]] = 1
  }
  return(adj)
}




adj2edge = function(adj, p) {
  edges = list()
  for(i in 1:p) {
    for(j in 1:p) {
      if (adj[i,j] == 1) {
        edges = c(edges, list(c(i,j)))
      }
    }
  }
  return(edges)
}




edge_diff_npvar = function(edges_star, adj_hat, p) {
  sum( abs( edge2adj(edges_star, p) - adj_hat ) )
}

