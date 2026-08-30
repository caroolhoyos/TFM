# Variante sin compactificar el ranking de los votantes

import random
import math
from statistics import median


def normalizar_ranking(ranking):
    # Convierte cada nivel de preferencia en una lista
    ranking_normalizado = []
    for grupos_votantes, preferencias in ranking:
        preferencias_norm = [
            nivel if isinstance(nivel, list) else [nivel]
            for nivel in preferencias
        ]
        ranking_normalizado.append((grupos_votantes, preferencias_norm))
    return ranking_normalizado


def obtener_candidatos(ranking):
    # Mantiene el orden de aparición y evita duplicados
    candidatos = []
    vistos = set()
    for _, preferencias in ranking:
        for nivel in preferencias:
            for c in nivel:
                if c not in vistos:
                    vistos.add(c)
                    candidatos.append(c)
    return candidatos


def obtener_profundidad_ranking(ranking):
    # Número máximo de niveles de preferencia presentes en un ranking individual
    return max((len(preferencias) for _, preferencias in ranking), default=0)


def precalcular_conteos_posiciones(ranking, candidatos, profundidad=None):
    # Cuenta cuántos votantes sitúan a cada candidato en cada posición original
    if profundidad is None:
        profundidad = obtener_profundidad_ranking(ranking)

    candidatos_set = set(candidatos)
    conteos = {c: [0] * profundidad for c in candidatos}
    for grupos_votantes, preferencias in ranking:
        for posicion, nivel in enumerate(preferencias):
            if posicion >= profundidad:
                break
            for c in nivel:
                if c in candidatos_set:
                    conteos[c][posicion] += grupos_votantes

    return conteos


def simular_ronda(Pi_r, ranking, candidatos, conteos_posiciones=None):
    # Calcula los puntos sin compactificar posiciones y elimina a quienes obtienen el mínimo
    if conteos_posiciones is None:
        conteos_posiciones = precalcular_conteos_posiciones(ranking, candidatos, len(Pi_r))

    puntuaciones = {
        c: sum(frecuencia * puntos for frecuencia, puntos in zip(conteos_posiciones.get(c, ()), Pi_r))
        for c in candidatos
    }

    minimo = min(puntuaciones.values())
    eliminados = [c for c, p in puntuaciones.items() if p == minimo]
    eliminados_set = set(eliminados)
    vivos = [c for c in candidatos if c not in eliminados_set]
    return eliminados, vivos, puntuaciones


def obtener_ganador(vivos, puntuaciones=None, resolver_por_puntuacion=False):
    # Devuelve el candidato ganador
    if len(vivos) == 1:
        return vivos[0]
    if resolver_por_puntuacion and vivos and puntuaciones:
        # En caso de quedar varios vivos elige al que obtenga mayor puntuación
        return max(vivos, key=puntuaciones.get)
    return None


def fitness(resultado, cand, max_rondas):
    # Función de fitness
    ganador, _, _, rondas_sobrevive, _ = resultado
    if ganador == cand and max_rondas > 0:
        # Premia las victorias en menos rondas
        return 1.0 + (max_rondas - rondas_sobrevive) / max_rondas

    return 0.0 if max_rondas == 0 else 0.8 * rondas_sobrevive / max_rondas


def inicializar_P(n, c, k, a, b):
    # Genera una población inicial de puntuaciones aleatorias ordenadas decrecientemente
    P = []
    for i in range(n):
        P_i = []
        for r in range(k):
            p_r = [random.randint(a, b) for _ in range(c)]
            p_r = sorted(p_r, reverse=True)
            P_i.append(p_r)
        P.append(P_i)
    return P


def pair_parents(population):
    # Mezcla la población y forma parejas de progenitores
    parents = []
    shuffled = population.copy()
    random.shuffle(shuffled)
    for i in range(0, len(shuffled), 2):
        if i + 1 < len(shuffled):
            parents.append((shuffled[i], shuffled[i + 1]))
        else:
            parents.append((shuffled[i], shuffled[0]))
    return parents


def uniform_crossover(parents):
    # Combina los valores de los vectores de ambos progenitores al azar
    children = []
    for p1, p2 in parents:
        child1 = []
        child2 = []
        for r in range(len(p1)):
            ronda1 = []
            ronda2 = []
            for j in range(len(p1[r])):
                if random.choice([True, False]):
                    ronda1.append(p1[r][j])
                    ronda2.append(p2[r][j])
                else:
                    ronda1.append(p2[r][j])
                    ronda2.append(p1[r][j])
            ronda1 = sorted(ronda1, reverse=True)
            ronda2 = sorted(ronda2, reverse=True)
            child1.append(ronda1)
            child2.append(ronda2)
        children.append(child1)
        children.append(child2)
    return children


def mutar_ronda(ronda, n_vivos, mutation_prob, sigma_efectiva, a, b):
    # Muta todas las posiciones, ya que en esta variante conservan su lugar original
    # Se mantiene n_vivos en la firma por compatibilidad con alg_gen.py
    n_posiciones_activas = len(ronda)
    hubo_mutacion = False
    if mutation_prob > 0:
        for j in range(n_posiciones_activas):
            if random.random() <= mutation_prob:
                delta = random.gauss(0, sigma_efectiva)
                nuevo = ronda[j] + delta
                ronda[j] = int(round(max(a, min(b, nuevo))))
                hubo_mutacion = True
    if hubo_mutacion:
        ronda[:n_posiciones_activas] = sorted(ronda[:n_posiciones_activas], reverse=True)
    ronda[n_posiciones_activas:] = [0] * (
        len(ronda) - n_posiciones_activas
    )


def simular_eleccion(Pi, ranking, cand, k, candidatos=None, conteos_posiciones=None):
    # Simula las rondas de las votaciones sin reasignar las posiciones eliminadas
    if candidatos is None:
        candidatos = obtener_candidatos(ranking)
    if conteos_posiciones is None:
        profundidad = max((len(ronda) for ronda in Pi), default=0)
        conteos_posiciones = precalcular_conteos_posiciones(ranking, candidatos, profundidad)

    vivos = candidatos[:]
    rondas_sobrevive = 0
    ultima_puntuacion = []

    if cand not in vivos:
        return None, vivos, False, rondas_sobrevive, ultima_puntuacion

    rondas_a_simular = min(k, len(Pi))

    for r in range(rondas_a_simular):
        if len(vivos) == 1:
            ganador = vivos[0]
            return ganador, vivos, cand in vivos, rondas_sobrevive, ultima_puntuacion

        Pi_r = Pi[r]
        eliminados, nuevos_vivos, puntuaciones = simular_ronda(Pi_r, ranking, vivos, conteos_posiciones)
        ultima_puntuacion = puntuaciones
        if cand in eliminados:
            ganador = nuevos_vivos[0] if len(nuevos_vivos) == 1 else None
            return ganador, nuevos_vivos, False, rondas_sobrevive, ultima_puntuacion

        rondas_sobrevive += 1
        vivos = nuevos_vivos[:]

        if len(vivos) == 0:
            return None, vivos, False, rondas_sobrevive, ultima_puntuacion

    if len(vivos) == 1:
        ganador = vivos[0]
        return ganador, vivos, cand in vivos, rondas_sobrevive, ultima_puntuacion

    ganador = obtener_ganador(vivos, ultima_puntuacion, resolver_por_puntuacion=True)

    return ganador, vivos, cand in vivos, rondas_sobrevive, ultima_puntuacion


def simular_eleccion_mutante(child, candidatos, cand, max_rondas, mutation_prob, sigma_efectiva, a, b, ranking, conteos_posiciones=None):
    # Simula las rondas de las votaciones y hace la mutación a la vez
    vivos = candidatos[:]
    rondas_sobrevive = 0
    ultima_puntuacion = {}

    if cand not in vivos:
        for ronda in child:
            ronda[:] = [0] * len(ronda)
        return None, vivos, False, 0, ultima_puntuacion

    if max_rondas == 0:
        ganador = obtener_ganador(vivos, resolver_por_puntuacion=False)
        for ronda in child:
            ronda[:] = [0] * len(ronda)
        return ganador, vivos, True, 0, ultima_puntuacion

    rondas_a_simular = min(max_rondas, len(child))

    for r in range(rondas_a_simular):
        ronda = child[r]

        if len(vivos) <= 1:
            ganador = obtener_ganador(vivos)

            for ronda_restante in child[r:]:
                ronda_restante[:] = [0] * len(ronda_restante)

            return (ganador, vivos, cand in vivos, rondas_sobrevive, ultima_puntuacion,)

        mutar_ronda(ronda, len(vivos), mutation_prob, sigma_efectiva, a, b)
        eliminados, nuevos_vivos, puntuaciones = simular_ronda(ronda, ranking, vivos, conteos_posiciones)
        ultima_puntuacion = puntuaciones

        if cand in eliminados:
            es_ultima_ronda = r + 1 == rondas_a_simular

            ganador = obtener_ganador(nuevos_vivos, puntuaciones, resolver_por_puntuacion=es_ultima_ronda)

            for ronda_restante in child[r + 1:]:
                ronda_restante[:] = [0] * len(ronda_restante)

            return (ganador, nuevos_vivos, False, rondas_sobrevive, ultima_puntuacion,)

        rondas_sobrevive += 1
        vivos = nuevos_vivos

        if not vivos:
            for ronda_restante in child[r + 1:]:
                ronda_restante[:] = [0] * len(ronda_restante)

            return None, vivos, False, rondas_sobrevive, ultima_puntuacion

        if len(vivos) == 1:
            ganador = vivos[0]

            for ronda_restante in child[r + 1:]:
                ronda_restante[:] = [0] * len(ronda_restante)

            return (ganador, vivos, cand in vivos, rondas_sobrevive, ultima_puntuacion,)

    ganador = obtener_ganador(vivos, ultima_puntuacion, resolver_por_puntuacion=True)

    for ronda_restante in child[rondas_a_simular:]:
        ronda_restante[:] = [0] * len(ronda_restante)

    return (ganador, vivos, cand in vivos, rondas_sobrevive, ultima_puntuacion,)


def mutate(children, mutation_prob, a, b, ranking, candidatos, cand, max_rondas, sigma=None, conteos_posiciones=None):
    # Muta los descendientes y evalúa el resultado electoral de cada uno
    sigma_efectiva = (b - a) * 0.5 if sigma is None else sigma
    resultados = [
        simular_eleccion_mutante(child, candidatos, cand, max_rondas, mutation_prob, sigma_efectiva, a, b, ranking, conteos_posiciones,)
        for child in children
    ]

    return children, resultados


def rank_selection(family, fitness_values, resultados, pop_size):
    # Selecciona individuos según su posición en el ranking
    indexed = sorted(range(len(family)), key=lambda i: fitness_values[i])
    weights = [0.0] * len(family)
    inicio = 0
    while inicio < len(indexed):
        fin = inicio + 1
        valor = fitness_values[indexed[inicio]]
        while fin < len(indexed) and fitness_values[indexed[fin]] == valor:
            fin += 1
        rango_medio = ((inicio + 1) + fin) / 2
        for posicion in range(inicio, fin):
            weights[indexed[posicion]] = rango_medio
        inicio = fin

    selected_idx = random.choices(range(len(family)), weights=weights, k=pop_size)
    new_pop = [family[i] for i in selected_idx]
    new_fitness = [fitness_values[i] for i in selected_idx]
    new_resultados = [resultados[i] for i in selected_idx]
    return new_pop, new_fitness, new_resultados


def roulette_selection(family, fitness_values, resultados, pop_size):
    # Asigna una probabilidad proporcional al fitness de cada individuo
    minimo = min(fitness_values)
    weights = [f - minimo + 1 for f in fitness_values]
    selected_idx = random.choices(range(len(family)), weights=weights, k=pop_size)
    new_pop = [family[i] for i in selected_idx]
    new_fitness = [fitness_values[i] for i in selected_idx]
    new_resultados = [resultados[i] for i in selected_idx]
    return new_pop, new_fitness, new_resultados


def seleccionar_poblacion(family, fitness_values, resultados, pop_size, selection_method):
    # Aplica el método de selección indicado
    if selection_method == "rank":
        return rank_selection(family, fitness_values, resultados, pop_size)
    if selection_method == "roulette":
        return roulette_selection(family, fitness_values, resultados, pop_size)
    raise ValueError("selection_method debe ser 'rank' o 'roulette'.")


def genetic_election(ranking, cand, generations, mutation_prob, pop_size, k=None, a=0, b=100, return_history=False, selection_method="rank"):

    # Inicializa el ranking, los conteos por posición y la población
    ranking = normalizar_ranking(ranking)
    candidatos = obtener_candidatos(ranking)
    numero_candidatos = len(candidatos)
    profundidad_ranking = obtener_profundidad_ranking(ranking)
    conteos_posiciones = precalcular_conteos_posiciones(ranking, candidatos, profundidad_ranking)

    if k is None:
        k = numero_candidatos - 1
    if k < 0:
        raise ValueError("k debe ser mayor o igual que 0.")

    max_rondas = max(min(k, numero_candidatos - 1), 0)

    population, population_resultados = mutate(inicializar_P(pop_size, profundidad_ranking, k, a, b), 0, a, b, ranking, candidatos, cand, max_rondas, conteos_posiciones=conteos_posiciones,)
    population_fitness = [fitness(resultado, cand, max_rondas) for resultado in population_resultados]

    best_final_fitness = -math.inf
    best_candidate = population[0]
    best_resultado = population_resultados[0]
    history = []

    for gen in range(generations):
        # Cruce, mutación y evaluación de fitness de cada generación
        parents = pair_parents(population)
        children = uniform_crossover(parents)

        sigma = (b - a) * 0.35 * (1 - gen / generations)
        mutated_pop, mutated_resultados = mutate(children, mutation_prob, a, b, ranking, candidatos, cand, max_rondas, sigma=sigma, conteos_posiciones=conteos_posiciones,)

        children_fitness = [fitness(r, cand, max_rondas) for r in mutated_resultados]

        family = population + mutated_pop
        family_resultados = population_resultados + mutated_resultados
        fitness_values = population_fitness + children_fitness

        best_pos_generation = max(range(len(fitness_values)), key=fitness_values.__getitem__)
        best_fitness_generation = fitness_values[best_pos_generation]

        elite_actualizado = best_fitness_generation > best_final_fitness
        if elite_actualizado:
            best_final_fitness = best_fitness_generation
            best_candidate = family[best_pos_generation]
            best_resultado = family_resultados[best_pos_generation]

        if return_history:
            history.append({"generation": gen + 1, "best_fitness": best_final_fitness, "generation_best_fitness": best_fitness_generation, "average_fitness": sum(fitness_values) / len(fitness_values), "median_fitness": median(fitness_values), })

        # Conserva el mejor individuo y selecciona el resto de la nueva población
        if elite_actualizado:
            family_for_selection = family[:best_pos_generation] + family[best_pos_generation + 1:]
            fitness_for_selection = fitness_values[:best_pos_generation] + fitness_values[best_pos_generation + 1:]
            resultados_for_selection = family_resultados[:best_pos_generation] + family_resultados[best_pos_generation + 1:]
        else:
            family_for_selection = family
            fitness_for_selection = fitness_values
            resultados_for_selection = family_resultados

        selected_pop, selected_fitness, selected_resultados = seleccionar_poblacion(family_for_selection, fitness_for_selection, resultados_for_selection, pop_size - 1, selection_method,)
        population = [best_candidate] + selected_pop
        population_fitness = [best_final_fitness] + selected_fitness
        population_resultados = [best_resultado] + selected_resultados

    resultado = best_resultado

    if return_history:
        return best_candidate, best_final_fitness, resultado, history
    return best_candidate, best_final_fitness, resultado
