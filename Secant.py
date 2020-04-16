def secant(f, x0, x1, N, verbose=True): 
    
    if verbose:
        print("*** Secant method ***")
    
    x2 = x1
                   
    for n in range(N):
        
        if verbose:
            print("x0 = ", x0)
            print("x1 = ", x1)
              
        deltaF = f(x1) - f(x0)
        
        if verbose:
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

        if verbose:
            print("f(x2)  = ", error)
       
        if (error == 0):  
            break 
  
        # update the value of interval  
        x0 = x1 
        x1 = x2 
          
    print("Root : ", x2)
    
    return x2
  

  

  