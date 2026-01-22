""" COSTE COMPUTACIONAL
    Mejor caso: O(n · log n)
    Peor caso: O(n^2) [El array ya esta ordenado o inversamente ordenado] Hay un desbalanceo importante
    Caso promedio: O(n· log n)

    El caso más ideal es que en cada división del array en dos sub-arrays 
    este mas o menos balanceado el tamaño del array de mayores y menos,
    por eso es importante el algoritmo de seleccion de pibote
 """
import time
data_unordered = [5, 3, 1, 2, 4, 6, 8, 7, 10, 9, 5, 5, 3, 1, 2, 1]

def quickSort(array):
    # Caso trivial que rompe la llamada recursiva
    if (len(array) <= 1):
        return array
    
    pibote = array[0]
    
    array_lower = []
    array_equal = []
    array_greater = []

    # Coste computacional "n" en cada llamada recursiva
    for num in array:
        if (num < pibote):
            array_lower.append(num)
        elif (num > pibote):
            array_greater.append(num)
        else:
            array_equal.append(num)
    
    # El problema grande se divide en dos subproblemas
    return [*quickSort(array_lower), *array_equal, *quickSort(array_greater)]


start = time.perf_counter()
print(quickSort(list([*data_unordered])))
end = time.perf_counter()
print(f"Duracion con pibote como primer elemento: {(end-start)*1000} ms")