import numpy as np

if __name__ == "__main__":
    points = (np.random.rand(3, 100) * 200 - 100) + (np.random.rand(3, 1) *  1000)
    mean = np.mean(points, axis=1)
    # print(mean.reshape(-1, 1).shape)
    X = points - mean.reshape(-1, 1)
    C = X @ X.T / 99
    A = np.linalg.cholesky(C).T
    T = np.linalg.inv(A.T) @ X
    


    print(np.cov(T))
