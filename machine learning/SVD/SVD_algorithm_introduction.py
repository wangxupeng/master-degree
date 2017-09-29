import numpy as np

if __name__ == '__main__':

    A = [4,0,3,-5]
    A = np.array(A).reshape(2,2)
    print("原始矩阵:\n",A)

    Am = np.dot(A.T,A)
    print("A的转置乘以A\n:",Am)

    e_value, e_vector = np.linalg.eig(Am)
    print("特征值\n:",e_value)
    sqrt_sigma = np.sqrt(e_value)
    print("根号sigma:\n",sqrt_sigma)
    S = np.diag(sqrt_sigma)
    S = np.linalg.inv(S)
    V=e_vector
    print("根号sigma的逆:\n",S)
    print("特征值V:\n", V)
    print("V的转置:\n",V.T)


    U = np.dot(np.dot(A,V),S)
    print("矩阵U:\n", U)

    A_bar = np.dot(np.dot(U,np.linalg.inv(S)),V.T)
    print("还原的矩阵A:\n",A_bar)
