# Técnica Voraz. Algoritmos voraces (V)

# Problema: Devolver cambio de monedas

# La aplicación de la técnica voraz es eficiente bajo algunos supuestos:
# ✓ Debe existir el valor unitario (1) en el conjunto de monedas,
#   pues en otro caso no será posible obtener todas las cantidades.
# ✓ Disponemos de una cantidad infinita de monedas.
# ✓ La "voracidad" no es eficiente en todos los sistemas monetarios.

# Por ejemplo: con el conjunto de monedas (1, 5, 11)
# ¿Cómo conseguir la cantidad C = 15?


def get_change(quantity, money_system = (1, 5, 11)):
    coin_counter_dic = { }
    coins = sorted(money_system, reverse=True)
    for coin in coins:
        coin_counter_dic[coin] = quantity // coin # optimización local con esperanza global.
        quantity = quantity % coin

        # Una vez procesamos una moneda la descartamos y continuamos
        # con otros candidatos

    return coin_counter_dic


print(get_change(55))