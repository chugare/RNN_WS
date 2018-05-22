

def g(n):
    for i in range(n):
        yield i


gk = g(5)
for i in range(100):
    print(next(gk))