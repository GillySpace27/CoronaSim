


def seperate(list, N):
    chunkSize = len(list)/N
    chunkSizeInt = int(chunkSize)
    #print(chunkSizeInt)
    chunks =  make_chunks(list, chunkSizeInt)
    while len(chunks) > N:
        NL = len(chunks) - 1
        chunks[NL - 1].extend(chunks.pop())
    return chunks
        
    
def make_chunks(list, n):
    n = max(1, n)
    return [list[i:i + n] for i in range(0, len(list), n)]

N = 8

list = []
for ii in range(157):
    list.append(ii)

#mychunks = chunks(list, n)

mychunks = seperate(list, N)

print("Chunks: " +str(len(mychunks)))

for chunk in mychunks:
    print(len(chunk))

