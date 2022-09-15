#! python3

### LIBRARIES
import math as mm

### FUNCTIONS

def fun(d, a):
    #### This is an implementation of a smooth max/min function
    ### alpha = 0     --> Arithmetic mean of elements of list d
    ### alpha >> 0    --> Smooth approximation of max(d)
    ### alpha << 0    --> Smooth approximation of min(d)
    ### see https://en.wikipedia.org/wiki/Smooth_maximum for more information
    
    sum_num = 0.0 
    sum_den = 0.0
    for dd in d:
        sum_num = sum_num + dd * mm.exp(a*dd)
        sum_den = sum_den + mm.exp(a*dd)
    
    return sum_num/sum_den

def gradientFun(d, a):
    ### This is the gradient of the function implemented in fun(d, a)
    
    S = fun(d, a)
    dS = []
    sum_den = 0.0
    for dd in d:
        sum_den = sum_den + mm.exp(a*dd)
    for dd in d:
        aux = mm.exp(a*dd) * (1 + a*(dd - S)) / sum_den
        dS.append( aux )
    
    return dS

def hessianFun(d, a):
    ### This is the hessian of the function implemented in fun(d, a)

    S = fun(d, a)
    dS = gradientFun(d, a)
    ddS = []

    sum_den = 0.0
    for dd in d:
        sum_den = sum_den + mm.exp(a*dd)

    for i in range(len(d)):
        ddS.append([])
        for j in range(len(d)):
            if i != j:
                aux1 = ( -a * mm.exp(a * d[i] ) * mm.exp( a * d[j] ) ) / ( sum_den ** 2 ) * ( 1 + a * ( d[i] - S)  )
                aux2 = ( mm.exp( a * d[i] ) ) / (sum_den) * ( a * -dS[j] )
            else:
                aux1 = ( a*mm.exp( a * d[i]) * sum_den - a * mm.exp(a * d[i] ) * mm.exp( a * d[i] ) ) / ( sum_den ** 2 ) * ( 1 + a * ( d[i] - S)  )
                aux2 = ( mm.exp( a * d[i] ) ) / (sum_den) * ( a * (1 - dS[j]) )
            ddS[i].append(aux1 + aux2)

    return ddS
    

def main():
    d1 = [1.0, 2.0, 1.0]
    d2 = [3.0, 2.0, -1.0, 6.0, -10.0, 20.0, -15.0, 13.0]

    a = 5.0

    print("max d1: ", fun(d1, a))
    print("max d2: ", fun(d2, a))

    print("min d1: ", fun(d1, -a))
    print("min d2: ", fun(d2, -a))

    print(gradientFun(d1, a))
    print(hessianFun(d1, a))

if __name__ == '__main__':
    main()
