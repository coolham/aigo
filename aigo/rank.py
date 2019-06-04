import time
def rank(key,a):
    n = len(a)
    for i in range(n):
        if key == a[i]:
            return i
    return -1



def rank1(key,a):
    lo = 0
    hi = len(a)-1
    while(lo<=hi):
        mid = lo + (hi-lo)/2
        mid = int(mid)
        if key<a[mid]:
            hi = mid-1
        elif key>a[mid]:
            lo = mid+1
        else:
            return mid
    return -1




start = time.clock()
s = [10,11,12,16,18,23,29,33,48,54,57,68,77,84,98]
m = []
for i in range(10000):
    m.append(i)
for i in range(0,100000):
    q = rank1(9000,m)

elapsed = (time.clock() - start)
print("Time used:", elapsed)
