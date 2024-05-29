import numpy as np
E1=[1,3,5,8,9]
b = len(E1)
E=np.zeros((b,1))
E[:b,0]=E1[:b]
print(E)

nmax= 10

dimensions = E.shape
rows, columns = dimensions
 
print("Rows:", rows)
print("Columns:", columns)

Emasked = np.zeros((b,nmax))
print(Emasked)

dimensions = Emasked.shape
rows, columns = dimensions
 
print("Rows:", rows)
print("Columns:", columns)

dtmp = np.arange(0,1,1)
print(dtmp)
dtmp = np.pad(dtmp, (0, b - len(dtmp)), 'constant')
print(dtmp)