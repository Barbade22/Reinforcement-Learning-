arr = [1,2,5,4,4,2,4,5,34,]
l = []
dub = []
for i in range(len(arr)):
    if arr[i] in l:
        dub.append(arr[i])
    else:
        l.append(arr[i])
        i +=1
print(dub)