


def seperate(list, N):
    chunkSize = len(list)/N
    chunkSizeInt = int(chunkSize) + 1
    print(chunkSizeInt)
    return make_chunks(list, chunkSizeInt)
    
def make_chunks(list, n):
    n = max(1, n)
    return [list[i:i + n] for i in range(0, len(list), n)]

n = 6
N = 28

list = []
for ii in range(1256):
    list.append(ii)

#mychunks = chunks(list, n)

mychunks = seperate(list, N)

for chunk in mychunks:
    print(len(chunk))

