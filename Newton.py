def newton(f, df, x0, N, verbose=True, debug=False):
    
    if verbose:
        print("*** Newton-Raphson method ***")
                  
    for i in range(N):
        
        derivative = df(x0)
        
        if derivative == 0:
            print("Error in newton : derivative is equal to 0")
            print("Stopping at the actual value.")
            break;
        
        x0 = x0 - f(x0)/derivative
        
        if verbose:
            print("Root = ", x0)
            
        if debug:
            print("Error = ", f(x0))
    
    return x0