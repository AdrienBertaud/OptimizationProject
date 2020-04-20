import matplotlib.pyplot as plt
import numpy as np

# source: https://www.codewithc.com/secant-method-matlab-program/

def secant(f, x0, x1, N, verbose=True, plot=False, debug=False):
    
    """
        Secant method is a numerical method to find the approximate root
        of polynomial equations. During the course of iteration, 
        this method assumes the function to be approximately linear 
        in the region of interest
        
        Advantages of Secant Method over other Root Finding Methods:
        - Its rate of convergence is more rapid than that of bisection method.
          So, secant method is considered to be a much faster root finding method.
        - In this method, there is no need to find the derivative of the function
          as in Newton-Raphson method.
        
        Limitations of Secant Method:
        The method fails to converge when f(xn) = f(xn-1)
        If X-axis is tangential to the curve, it may not converge to the solution.
    """
    
    if plot:
        x = []
        y = []
        x.append(x0)
        y.append(f(x0))
        x.append(x1)
        y.append(f(x1))
    
    if verbose:
        print("*** Secant method ***")
            
    # Initialization of the n + 2 value.
    x2 = x1
                   
    for n in range(N):
        
        if debug:
            print("x0 = ", x0)
            print("x1 = ", x1)
            
        fx0 = f(x0)
        fx1 = f(x1)
              
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
        
        if plot:
            x.append(x2)
            y.append(error)

        if debug:
            print("f(x2)  = ", error)
              
        # update the value of interval  
        x0 = x1 
        x1 = x2 
    
    if debug:
        print("x : ", x)
        print("y : ", y)
    
    if plot:
        plt.figure()
        plt.plot(x, y, 'bx')
        
        minX = min(x)
        maxX = max(x)
        stepX = (maxX-minX)/100
        
        if debug:
            print("minX = ", minX)
            print("maxX = ", maxX)
        
        t = np.arange(minX, maxX, stepX)
        plt.plot(t, f(t), 'grey')
        
        for i in range(len(x)):
            #plt.annotate(str(i), (x[i], y[i]))
            plt.text(x[i], y[i], str(i), {'color': 'red', 'fontsize': 14})
            
        plt.show()
    
    return x2
  

  

  