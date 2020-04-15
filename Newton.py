def newton(f, df, x0, N):
    
    print("*** Newton-Raphson method ***")
          
    for i in range(N):
        x0 = x0 - f(x0)/df(x0)
        print("Racine = ", x0)
        print("Erreur = ", f(x0))
        