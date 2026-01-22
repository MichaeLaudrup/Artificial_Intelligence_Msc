def fibonnaci(n, cache = {0: 0, 1:1}):
    if(n in cache):
        return cache[n]
    else:
        cache[n] = fibonnaci(n-1, cache) + fibonnaci(n-2, cache)
        return cache[n]

print(fibonnaci(10))