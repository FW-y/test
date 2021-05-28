from scipy.optimize import minimize
import numpy as np
import math
 
"""
                       [0.5,0.3,0.2]
求解信道传递矩阵为P(Y|X)=[0.3,0.5,0.2]的非对称信道的信道容量C
                       [0.1,0.2,0.7]
--------------------------------------------------------
                          [0.5,0.3,0.2]
P(Y)=P(X)P(Y|X)=[p1,p2,p3][0.3,0.5,0.2]=[0.5p1+0.3p2+0.1p3,0.3p1+0.5p2+0.2p3,0.2p1+0.2p2+0.7p3]
                          [0.1,0.2,0.7]
-----------------------------------------------------------------------------------------------
H(Y|X)=P(X)H(Y|X=x)=H(0.5,0.3,0.2)p1+H(0.3,0.5,0.2)p2+H(0.1,0.2,0.7)p3
----------------------------------------------------------------------
C=max{I(X;Y)}=H(Y)-H(Y|X)=H(0.5p1+0.3p2+0.1p3,0.3p1+0.5p2+0.2p3,0.2p1+0.2p2+0.7p3)-
  P(x)                    (H(0.5,0.3,0.2)p1+H(0.3,0.5,0.2)p2+H(0.1,0.2,0.7)p3)=f(p1,p2,p3)
------------------------------------------------------------------------------------------
原问题等效为max f(p1,p2,p3)
            s.t.
              p1+p2+p3=1
              0<pi<1 i=1,2,3
----------------------------
"""
 
 
def entropy(x):
 
    return -x * math.log(x, 2)
 
 
def fun(p):
 
    a = entropy((0.5*p[0]+0.3*p[1]+0.1*p[2]))
    b = entropy((0.3*p[0]+0.5*p[1]+0.2*p[2]))
    c = entropy((0.2*p[0]+0.2*p[1]+0.7*p[2]))
    d = entropy(0.5) + entropy(0.3) + entropy(0.2)
    e = entropy(0.1) + entropy(0.2) + entropy(0.7)
 
    return -(a + b + c - d * (p[0] + p[1]) - e * p[2])
 
 
if __name__ == '__main__':
 
    p0 = np.array([1/3., 1/3., 1/3.])  # 初始信源分布
    I0 = -fun(p0)                      # 初始互信息
 
    # 约束条件 等式约束p1+p2+p3=1
    cons = ({'type': 'eq',
             'fun': lambda p: np.array([p[0] + p[1] + p[2] - 1]),
             'jac': lambda p: np.array([1, 1, 1])})
 
    # 变量的界 0<pi<1
    bnd = ((0, 1), (0, 1), (0, 1))
 
    # 最优化方法Sequential Least SQuares Programming optimization algorithm 误差界10^-14
    res = minimize(fun, p0, method='SLSQP', constraints=cons, bounds=bnd, tol=1.e-14)
 
    print("初始信源分布为均匀分布", p0)
    print("初始互信息I(X;Y)=", I0)
    print(res.success)
    print("信道容量C=max{I(X;Y)}=", -res.fun)
    print("互信息达到最大时的信源分布为", res.x)