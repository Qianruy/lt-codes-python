import mpmath as mp        

k = 3          
delta = 0.0  

g = lambda lam: k*lam + mp.e**(-lam)*((k-1)*lam + k*(1-delta)) - k
         
lam = mp.findroot(g, k)   
print("λ =", lam)
v2 = 1 - mp.e**(-lam)*(1-delta)
print(f"v2 = {float(v2):.6f}")
c_star = lam / (k*v2**(k-1))
print("c* =", c_star)
print(f"alpha* = {(1/c_star-1)*100}(%)")