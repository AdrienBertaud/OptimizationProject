def quasiNewton(f, df, x0, N, verbose=True, debug=False):
    
    # Use this alpha for every line search
    alpha = np.linspace(0.1,1.0,N)
    
    # Initialize delta_xq and gamma
    delta_xq = np.zeros((2,1))
    gamma = np.zeros((2,1))
    part1 = np.zeros((2,2))
    part2 = np.zeros((2,2))
    part3 = np.zeros((2,2))
    part4 = np.zeros((2,2))
    part5 = np.zeros((2,2))
    part6 = np.zeros((2,1))
    part7 = np.zeros((1,1))
    part8 = np.zeros((2,2))
    part9 = np.zeros((2,2))
    
    # Initialize xq
    xq = np.zeros((N+1,2))
    xq[0] = x_start
    
    # Initialize gradient storage
    g = np.zeros((N+1,2))
    g[0] = dfdx(xq[0])
    
    # Initialize hessian storage
    h = np.zeros((N+1,2,2))
    h[0] = [[1, 0.0],[0.0, 1]]
    for i in range(N):
        
        # Compute search direction and magnitude (dx)
        #  with dx = -alpha * inv(h) * grad
        delta_xq = -np.dot(alpha[i],np.linalg.solve(h[i],g[i]))
        xq[i+1] = xq[i] + delta_xq
    
        # Get gradient update for next step
        g[i+1] = dfdx(xq[i+1])
    
        # Get hessian update for next step
        gamma = g[i+1]-g[i]
        part1 = np.outer(gamma,gamma)
        part2 = np.outer(gamma,delta_xq)
        part3 = np.dot(np.linalg.pinv(part2),part1)
    
        part4 = np.outer(delta_xq,delta_xq)
        part5 = np.dot(h[i],part4)
        part6 = np.dot(part5,h[i])
        part7 = np.dot(delta_xq,h[i])
        part8 = np.dot(part7,delta_xq)
        part9 = np.dot(part6,1/part8)
        
        h[i+1] = h[i] + part3 - part9
