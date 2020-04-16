def secant(f, x0, x1, N, verbose=True, debug=False): 
    
    if verbose:
        print("*** Secant method ***")
            
    x2 = x1
                   
    for n in range(N):
        
        if debug:
            print("x0 = ", x0)
            print("x1 = ", x1)
            
        fx0 = f(x0)
        fx1 = f(x1)
        
        if(fx0 < 0 or fx0 < 0):
            print("f(x) < 0, stopping iteration")
            break
              
        deltaF = fx1 - fx0
        
        if debug:
            print("deltaF = ", deltaF)
        
        if(deltaF == 0):
            print("Error : f(x1) = f(x0)")
            print("Stopping at the actual value.")
            break;
        
        # calculate the intermediate value  
        x2 = x1 - f(x1) * (x1 - x0 ) / deltaF
        
        if verbose:
            print("x2 = ", x2)

        error = f(x2)        

        if debug:
            print("f(x2)  = ", error)
            
        if(error < 0):
            print("f(x2) <= 0, stopping at the closest value.")
            
            if(abs(f(x1)) < abs(error)):
                x2 = x1
                print("x2 = ", x2)
                
            break
  
        # update the value of interval  
        x0 = x1 
        x1 = x2 
              
    return x2
  

  

  