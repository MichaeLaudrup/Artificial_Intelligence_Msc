# Dada dos listas encontrar sus elementos comunes
def common_elements(list1: list, list2: list) -> list:
    result = [ el for el in list1 if el in list2]
    return result

common_elements([1,2,6,7,12,13,15],[2,3,4,7,13])