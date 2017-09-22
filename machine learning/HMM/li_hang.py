#coding=utf8  

import numpy as np  
  
class HMM(object):  
    def __init__(self, A, B, pi):  
        ''''' 
        A: 状态转移概率矩阵 
        B: 输出观察概率矩阵 
        pi: 初始化状态向量 
        '''  
        self.A = np.array(A)  
        self.B = np.array(B)  
        self.pi = np.array(pi)  
        self.N = self.A.shape[0]    # 总共状态个数  
        self.M = self.B.shape[1]    # 总共观察值个数     
        
      
    # 输出HMM的参数信息  
    def printHMM(self):  
        print ("==================================================")  
        print ("HMM content: N =",self.N,",M =",self.M)  
        for i in range(self.N):  
            if i==0:  
                print ("hmm.A ",self.A[i,:]," hmm.B ",self.B[i,:])  
            else:  
                print ("      ",self.A[i,:],"       ",self.B[i,:])  
        print ("hmm.pi",self.pi)  
        print ("==================================================")  
                      
      
    # 前向算法    
    def forwar(self, T, O, alpha, prob):  
        ''''' 
        T: 观察序列的长度 
        O: 观察序列 
        alpha: 运算中用到的临时数组 
        prob: 返回值所要求的概率 
        '''      
          
        # 初始化  
        for i in range(self.N):  
            alpha[0, i] = self.pi[i] * self.B[i, O[0]]  
  
        # 递归  
        for t in range(T-1):  
            for j in range(self.N):  
                sum = 0.0  
                for i in range(self.N):  
                    sum += alpha[t, i] * self.A[i, j]  
                alpha[t+1, j] = sum * self.B[j, O[t+1]]          
          
        # 终止  
        sum = 0.0  
        for i in range(self.N):  
            sum += alpha[T-1, i]  
          
        prob[0] *= sum     
  
      
    # 带修正的前向算法  
    def forwardWithScale(self, T, O, alpha, scale, prob):  
        scale[0] = 0.0  
          
        # 初始化  
        for i in range(self.N):  
            alpha[0, i] = self.pi[i] * self.B[i, O[0]]  
            scale[0] += alpha[0, i]  
              
        for i in range(self.N):  
            alpha[0, i] /= scale[0]  
          
        # 递归  
        for t in range(T-1):  
            scale[t+1] = 0.0  
            for j in range(self.N):  
                sum = 0.0  
                for i in range(self.N):  
                    sum += alpha[t, i] * self.A[i, j]  
                  
                alpha[t+1, j] = sum * self.B[j, O[t+1]]  
                scale[t+1] += alpha[t+1, j]  
              
            for j in range(self.N):  
                alpha[t+1, j] /= scale[t+1]  
           
        # 终止  
        for t in range(T):  
            prob[0] += np.log(scale[t])         
              
              
    def back(self, T, O, beta, prob):    
        ''''' 
        T: 观察序列的长度    len(O) 
        O: 观察序列 
        beta: 计算时用到的临时数组 
        prob: 返回值；所要求的概率 
        '''   
          
        # 初始化                 
        for i in range(self.N):  
            beta[T-1, i] = 1.0  
          
        # 递归  
        for t in range(T-2, -1, -1): # 从T-2开始递减；即T-2, T-3, T-4, ..., 0  
            for i in range(self.N):  
                sum = 0.0  
                for j in range(self.N):  
                    sum += self.A[i, j] * self.B[j, O[t+1]] * beta[t+1, j]  
                  
                beta[t, i] = sum  
         
        # 终止  
        sum = 0.0  
        for i in range(self.N):  
            sum +=  self.pi[i]*self.B[i,O[0]]*beta[0,i]  
          
        prob[0] = sum      
          
          
    # 带修正的后向算法  
    def backwardWithScale(self, T, O, beta, scale):  
        ''''' 
        T: 观察序列的长度 len(O) 
        O: 观察序列 
        beta: 计算时用到的临时数组 
        '''  
        # 初始化  
        for i in range(self.N):  
            beta[T-1, i] = 1.0  
          
        # 递归                 
        for t in range(T-2, -1, -1):  
            for i in range(self.N):  
                sum = 0.0  
                for j in range(self.N):  
                    sum += self.A[i, j] * self.B[j, O[t+1]] * beta[t+1, j]  
                  
                beta[t, i] = sum / scale[t+1]         
                  
      
    # viterbi算法              
    def viterbi(self, O):  
        ''''' 
        O: 观察序列 
        '''  
        T = len(O)  
        # 初始化  
        delta = np.zeros((T, self.N), np.float)  
        phi = np.zeros((T, self.N), np.float)  
        I = np.zeros(T)  
          
        for i in range(self.N):  
            delta[0, i] = self.pi[i] * self.B[i, O[0]]  
            phi[0, i] = 0.0  
          
        # 递归  
        for t in range(1, T):  
            for i in range(self.N):  
                delta[t, i] = self.B[i, O[t]] * np.array([delta[t-1, j] * self.A[j, i] for j in range(self.N)] ).max()  
                phi = np.array([delta[t-1, j] * self.A[j, i] for j in range(self.N)]).argmax()  
              
        # 终止  
        prob = delta[T-1, :].max()  
        I[T-1] = delta[T-1, :].argmax()  
          
        for t in range(T-2, -1, -1):  
            I[t] = phi[I[t+1]]  
              
          
        return prob, I  
      
      
    # 计算gamma(计算A所需的分母；详情见李航的统计学习) : 时刻t时马尔可夫链处于状态Si的概率  
    def computeGamma(self, T, alpha, beta, gamma):  
        ''''''''  
        for t in range(T):  
            for i in range(self.N):  
                sum = 0.0  
                for j in range(self.N):  
                    sum += alpha[t, j] * beta[t, j]  
                  
                gamma[t, i] = (alpha[t, i] * beta[t, i]) / sum     
      
    # 计算sai(i,j)(计算A所需的分子) 为给定训练序列O和模型lambda时  
    def computeXi(self, T, O, alpha, beta, Xi):  
          
        for t in range(T-1):  
            sum = 0.0  
            for i in range(self.N):  
                for j in range(self.N):  
                    Xi[t, i, j] = alpha[t, i] * self.A[i, j] * self.B[j, O[t+1]] * beta[t+1, j]  
                    sum += Xi[t, i, j]  
              
            for i in range(self.N):  
                for j in range(self.N):  
                    Xi[t, i, j] /= sum  
     
      
    #  输入 L个观察序列O，初始模型：HMM={A,B,pi,N,M}  
    def BaumWelch(self, L, T, O, alpha, beta, gamma):                                      
        DELTA = 0.01 ; round = 0 ; flag = 1 ; probf = [0.0]  
        delta = 0.0; probprev = 0.0 ; ratio = 0.0 ; deltaprev = 10e-70  
          
        xi = np.zeros((T, self.N, self.N)) # 计算A的分子  
        pi = np.zeros((T), np.float)    # 状态初始化概率  
          
        denominatorA = np.zeros((self.N), np.float) # 辅助计算A的分母的变量  
        denominatorB = np.zeros((self.N), np.float)  
        numeratorA = np.zeros((self.N, self.N), np.float)   # 辅助计算A的分子的变量  
        numeratorB = np.zeros((self.N, self.M), np.float)   # 针对输出观察概率矩阵  
        scale = np.zeros((T), np.float)  
          
        while True:  
            probf[0] =0  
              
            # E_step  
            for l in range(L):  
                self.forwardWithScale(T, O[l], alpha, scale, probf)  
                self.backwardWithScale(T, O[l], beta, scale)  
                self.computeGamma(T, alpha, beta, gamma)    # (t, i)  
                self.computeXi(T, O[l], alpha, beta, xi)    #(t, i, j)  
                  
                for i in range(self.N):  
                    pi[i] += gamma[0, i]  
                    for t in range(T-1):  
                        denominatorA[i] += gamma[t, i]  
                        denominatorB[i] += gamma[t, i]  
                    denominatorB[i] += gamma[T-1, i]  
                  
                    for j in range(self.N):  
                        for t in range(T-1):  
                            numeratorA[i, j] += xi[t, i, j]  
                          
                    for k in range(self.M): # M为观察状态取值个数  
                        for t in range(T):  
                            if O[l][t] == k:  
                                numeratorB[i, k] += gamma[t, i]      
                                  
              
            # M_step。 计算pi, A, B  
            for i in range(self.N): # 这个for循环也可以放到for l in range(L)里面  
                self.pi[i] = 0.001 / self.N + 0.999 * pi[i] / L  
                  
                for j in range(self.N):  
                    self.A[i, j] = 0.001 / self.N + 0.999 * numeratorA[i, j] / denominatorA[i]                      
                    numeratorA[i, j] = 0.0  
                  
                for k in range(self.M):  
                    self.B[i, k] = 0.001 / self.N + 0.999 * numeratorB[i, k] / denominatorB[i]  
                    numeratorB[i, k] = 0.0     
                  
                #重置  
                pi[i] = denominatorA[i] = denominatorB[i] = 0.0  
                  
            if flag == 1:  
                flag = 0  
                probprev = probf[0]  
                ratio = 1  
                continue  
              
            delta = probf[0] -  probprev   
            ratio = delta / deltaprev     
            probprev = probf[0]  
            deltaprev = delta  
            round += 1  
              
            if ratio <= DELTA :  
                print('num iteration: ', round)     
                break  
          
  
if __name__ == '__main__':  
    print ("python my HMM")  
      
    # 初始的状态概率矩阵pi；状态转移矩阵A；输出观察概率矩阵B; 观察序列  
    pi = [0.5,0.5]  
    A = [[0.8125,0.1875],[0.2,0.8]]  
    B = [[0.875,0.125],[0.25,0.75]]  
    O = [  
         [1,0,0,1,1,0,0,0,0],  
         [1,1,0,1,0,0,1,1,0],  
         [0,0,1,1,0,0,1,1,1]  
        ]  
    L = len(O)  
    T = len(O[0])   #  T等于最长序列的长度就好了  
      
    hmm = HMM(A, B, pi)  
    alpha = np.zeros((T,hmm.N),np.float)  
    beta = np.zeros((T,hmm.N),np.float)  
    gamma = np.zeros((T,hmm.N),np.float)  
      
    # 训练  
    hmm.BaumWelch(L,T,O,alpha,beta,gamma)  
      
    # 输出HMM参数信息  
    hmm.printHMM()      
