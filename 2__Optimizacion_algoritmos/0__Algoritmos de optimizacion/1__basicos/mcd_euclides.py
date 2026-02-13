import time
import numpy as np

random_n = np.random.randint(10000000)
random_m = np.random.randint(10000000)

print(f"Minimo comun divisor entre {random_n} y {random_m} es: ")
# Calcular el maximo comun divisor siguiendo el algoritmo de euclides
# es una manera mas eficiente y elegante de resolver el algoritmo
# sobretodo para numeros grandes o programacion

def rest_mcd(n,m):
    while(m > 0):
        if n > m:
            n = n - m
        else:
            m = m - n
    return n

start = time.perf_counter() * 1000000
res = rest_mcd(random_n, random_m)
end = time.perf_counter() * 1000000
first_algorithm_time = end - start
print(f"Resultado: {res}\nTiempo calculo mcd euclides con resta {first_algorithm_time:.2f}us")

# Version avanzada del algoritmo de euclides
def mcd(m, n):
    while(n != 0):
        res = m % n
        m = n
        n = res
    return m

start = time.perf_counter() * 1000000
res = mcd(random_n, random_m)
end = time.perf_counter() * 1000000
second_algorithm_time = end-start
print(f"Resultado: {res}\nTiempo calculo mcd euclides con division {second_algorithm_time:.2f}us")

print(f"El segundo algoritmo es un {(second_algorithm_time / first_algorithm_time * 100):.2f}% mas rapido")
        