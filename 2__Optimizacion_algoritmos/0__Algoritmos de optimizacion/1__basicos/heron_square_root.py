# Calcular la raiz cuadrada de un numero de manera programatica e iterativa
# siguiendo el metodo de heron

def heron_square_root(n, x):
    next_x = 0.5 * (x + (n / x))
    if abs(x - next_x) < 1e-3:
        return x
    else:
        return heron_square_root(n, next_x)

print(heron_square_root(100, 1))