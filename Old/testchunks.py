import numpy as np


def seperate(list, N):
    chunkSize = len(list)/N
    chunkSizeInt = int(chunkSize)
    remainder = int((chunkSize % chunkSizeInt) * N)
    
    chunks = [ [] for _ in range(N)] 
    for proc in np.arange(N):
        thisLen = chunkSizeInt
        if remainder > 0:
            thisLen += 1
            remainder -= 1
        for nn in np.arange(thisLen):
            chunks[proc].extend([list.pop(0)])
    return chunks
        
    
def make_chunks(list, n):
    n = max(1, n)
    return [list[i:i + n] for i in range(0, len(list), n)]

N = 8

list = []
for ii in range(146):
    list.append(ii)

#print(len(list))
#for li in range(10):
#    a = list.pop(0)
#    print(a)

#mychunks = chunks(list, n)

mychunks = seperate(list, N)
print(mychunks[0])
print("Chunks: " +str(len(mychunks)))

for chunk in mychunks:
    print(len(chunk))

