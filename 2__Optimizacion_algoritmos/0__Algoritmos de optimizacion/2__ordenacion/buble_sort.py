import time
import random

"""
   Tanto mejor caso como peor caso tiene una complejidad O(n^2), excepto si añadimos mecanismo de
   control que termine el algoritmo cuando no hay intercambios en una iteración, en este caso
   tiene coste O(n) para caso más favorables

   No es eficiente, pero muy facil de implementar
"""
array = random.sample(range(1,1000), 20)

"""
   Esta estrategia siempre recorre de inicio a fin el array original, no tiene
   en cuenta que en la iteración i-esima del bucle while no es necesario revisar las
   ultimas "i" posiciones sino que se finaliza cuando no se realizan cambios
"""
def buble_sort(array):
    while(True):
        changes = 0
        for idx in range(len(array) - 1):
            if(array[idx] > array[idx + 1]):
                changes += 1
                aux = array[idx]
                array[idx] = array[idx + 1]
                array[idx + 1] = aux
        if(changes == 0): 
            return array

start = time.perf_counter() * 1000000
res = buble_sort(array)
end = time.perf_counter() * 1000000
first_algorithm_time = end - start

"""
   Este método es una optimización del algoritmo de burbuja tradicional. A diferencia del método
   buble_sort, que siempre recorre todo el array, buble_sort_less_iterations reduce el número de
   iteraciones necesarias al no revisar las últimas posiciones ya ordenadas en cada pasada. 
   Esto se logra mediante el uso de una variable endPos que disminuye con cada iteración del bucle
   externo. Si en una pasada completa no se realizan cambios, el algoritmo termina, indicando que
   el array ya está ordenado. Esta mejora puede reducir el tiempo de ejecución en casos donde el
   array está parcialmente ordenado.
"""
def buble_sort_less_iterations(array):
    for endPos in range(len(array) - 1, 0, -1):
        changes = 0
        for idx in range(endPos):
            if(array[idx] > array[idx + 1]):
                changes += 1
                aux = array[idx]
                array[idx] = array[idx + 1]
                array[idx + 1] = aux
        if (changes == 0): return array

start = time.perf_counter() * 1000000
res = buble_sort_less_iterations(array)
end = time.perf_counter() * 1000000
second_algorithm_time = end - start

print(f"Array original: {array}")
print(f"Resultado del primer algoritmo: {res}")
print(f"Tiempo del primer algoritmo: {first_algorithm_time} microsegundos")
print(f"Resultado del segundo algoritmo: {res}")
print(f"Tiempo del segundo algoritmo: {second_algorithm_time} microsegundos")
if first_algorithm_time > second_algorithm_time:
    improvement = first_algorithm_time - second_algorithm_time
    percentage_improvement = (improvement / first_algorithm_time) * 100
    print(f"El segundo algoritmo es {improvement} microsegundos más rápido que el primero")
    print(f"El segundo algoritmo es un {percentage_improvement:.2f}% más rápido que el primero")
else:
    print("El segundo algoritmo no es más rápido que el primero")
