import random
def RandomSeq(n,a,b):
    e = []
    for i in range(n):
        q = random.randint(a,b)
        if q not in e:
            e.append(q)
    return e



def Average(a):
    sum = 0
    count = 0
    for i in a:
        sum+=i
        count+=1
    q = sum/count
    return q



w = RandomSeq(2,1,10)
print(w)
z =  Average(w)
print(z)