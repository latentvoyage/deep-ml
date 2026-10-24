import numpy as np
def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:


    a = np.array(a)
    b = np.array(b)
    if a.shape[-1] != b.shape[-2]:
        return -1
    c = np.zeros((a.shape[-2], b.shape[-1]))
    # print(c.shape)
    for i in range(c.shape[-2]):
        for j in range(c.shape[-1]):
            c[i][j] = np.dot(a[i,:], b[:,j])
    return c