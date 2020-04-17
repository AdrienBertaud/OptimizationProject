import Cubic
import Newton
import Secant
from importlib import reload
reload(Cubic)
reload(Newton)
reload(Secant)
from Newton import newton
from Secant import secant

nbIter = 10

x0 = 2.5 # Random initialization

newton(Cubic.f, Cubic.df, x0, nbIter)

# initializing the values  .
x1 = 2;  
secant(Cubic.f, x0, x1, nbIter);
