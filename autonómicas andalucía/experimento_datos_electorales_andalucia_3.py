# Experimento de P8 con perfil reducido y alg_gen_men

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
from statistics import mean, median
import sys
import time

SRC_DIR = Path(__file__).resolve().parent
MODULOS_DIR = SRC_DIR.parent
if str(MODULOS_DIR) not in sys.path:
    sys.path.insert(0, str(MODULOS_DIR))

from algoritmos_genéticos.alg_gen_men import genetic_election
from algoritmos_genéticos.graficos_alg_gen import crear_graficos_convergencia



CONFIGURACION_GENETICO = {
    "ejecuciones": 3,
    "generations": 150,
    "pop_size": 50,
    "mutation_prob": 0.10,
    "k": None,  # por defecto: número de candidatos - 1
    "selection_method": "rank",  # rank o roulette
    "a": 0,
    "b": 10,
    "seed": 20260731,
}

RAIZ_PROYECTO = SRC_DIR.parent
CSV_DATOS = (
    RAIZ_PROYECTO
    / "datos_electorales_andalucia_26"
    / "MD3558"
    / "3558_num.csv"
)
OUTPUT_DIR = SRC_DIR / "resultados 3"

# Relaciona cada líder de P8 con su formación política
PARTIDOS = {
    "PP": "LIDERES_1: Juan Manuel Moreno",
    "PSOE": "LIDERES_2: María Jesús Montero",
    "VOX": "LIDERES_3: Manuel Gavira",
    "Por Andalucía": "LIDERES_4: Antonio Maíllo",
    "Adelante Andalucía": "LIDERES_5: José Ignacio García",
}

# Partidos que se conservan en el perfil reducido
PARTIDOS_NACIONALES = list(PARTIDOS)

_NACIONALES_SET = set(PARTIDOS_NACIONALES)
PARTIDOS_TERRITORIALES = [p for p in PARTIDOS if p not in _NACIONALES_SET]

PARTIDOS_OBJETIVO = PARTIDOS_NACIONALES

# Trata los códigos 97, 98 y 99 como una puntuación de 0
VALOR_SIN_RESPUESTA = ""
VALORES_PUNTUACION_CERO = {"97", "98", "99"}
PUNTUACION_MINIMA = 0
PUNTUACION_MAXIMA = 10


def ranking_participante(fila: dict[str, str]) -> list[list[str]]:
    # Convierte las valoraciones de la encuesta en un ranking por niveles
    por_puntuacion: dict[int, list[str]] = {
        puntuacion: [] for puntuacion in range(PUNTUACION_MAXIMA, PUNTUACION_MINIMA - 1, -1)
    }
    hay_respuesta = False
    for partido, variable in PARTIDOS.items():
        valor = fila[variable].strip()
        if valor == VALOR_SIN_RESPUESTA:
            continue
        if valor in VALORES_PUNTUACION_CERO:
            puntuacion = 0
        else:
            puntuacion = int(valor)
            if not PUNTUACION_MINIMA <= puntuacion <= PUNTUACION_MAXIMA:
                raise ValueError(f"Valor inesperado en {variable}: {valor!r}")
        por_puntuacion[puntuacion].append(partido)
        hay_respuesta = True

    if not hay_respuesta:
        return []

    return [
        por_puntuacion[p]
        for p in range(PUNTUACION_MAXIMA, PUNTUACION_MINIMA - 1, -1)
        if por_puntuacion[p]
    ]


def reducir_a_nacionales(ranking: list[list[str]]) -> list[list[str]]:
    # Conserva solo los partidos seleccionados y compacta el ranking
    reducido = []
    for nivel in ranking:
        nivel_nacional = [p for p in nivel if p in _NACIONALES_SET]
        if nivel_nacional:
            reducido.append(nivel_nacional)
    return reducido


def cargar_rankings(csv_path: Path = CSV_DATOS):
    # Carga y agrupa los rankings de los participantes
    participantes = []
    conteo = Counter()

    with csv_path.open(encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo, delimiter=";")
        for fila in lector:
            ranking = reducir_a_nacionales(ranking_participante(fila))
            participantes.append( { "registro": fila["REGISTRO: Número de registro"], "cuestionario": fila["CUES: Cuestionario"], "ranking": ranking, } )
            if ranking:
                clave = tuple(tuple(nivel) for nivel in ranking)
                conteo[clave] += 1

    perfil = [
        (frecuencia, [list(nivel) for nivel in ranking])
        for ranking, frecuencia in conteo.most_common()
    ]
    return participantes, perfil


def resolver_partido(texto: str) -> str:
    # Valida y normaliza el nombre del partido
    coincidencias = [p for p in PARTIDOS_NACIONALES if p.casefold() == texto.casefold()]
    if not coincidencias:
        disponibles = ", ".join(PARTIDOS_NACIONALES)
        raise argparse.ArgumentTypeError( f"Partido desconocido o no conservado: {texto!r}. Opciones: {disponibles}" )
    return coincidencias[0]


def guardar_rankings(participantes, output_path: Path):
    # Guarda los rankings individuales en CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as archivo:
        campos = ["registro", "cuestionario", "num_partidos", "num_niveles", "ranking"]
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        for participante in participantes:
            ranking = participante["ranking"]
            writer.writerow( { "registro": participante["registro"], "cuestionario": participante["cuestionario"], "num_partidos": sum(map(len, ranking)), "num_niveles": len(ranking), "ranking": json.dumps(ranking, ensure_ascii=False), } )


def ejecutar_genetico(perfil, partido, args, ejecucion, partido_idx):
    # Ejecuta el algoritmo genético para un partido
    seed = args.seed + partido_idx * 100_000 + ejecucion - 1
    random.seed(seed)
    inicio = time.perf_counter()
    puntuacion, fitness, resultado, history = genetic_election( ranking=perfil, cand=partido, generations=args.generations, mutation_prob=args.mutation_prob, pop_size=args.pop_size, k=args.k, a=args.a, b=args.b, return_history=True, selection_method=args.selection_method, )
    duracion = time.perf_counter() - inicio
    ganador, vivos, cand_vivo, rondas, ultima_puntuacion, ronda_ganadora = resultado

    nombre_partido = (
        partido.casefold() .replace(" ", "_") .replace("/", "_") .replace("-", "_")
    )
    directorio = (
        args.output_dir
        / "graficos"
        / nombre_partido
        / f"ejecucion_{ejecucion:02d}"
    )
    directorio.mkdir(parents=True, exist_ok=True)
    graficos = crear_graficos_convergencia(history, directorio)
    (directorio / "historial.csv").write_text( "generation,best_fitness,generation_best_fitness,average_fitness\n" + "\n".join( f"{h['generation']},{h['best_fitness']},{h['generation_best_fitness']},{h['average_fitness']}" for h in history ) + "\n", encoding="utf-8", )
    return {
        "ejecucion": ejecucion,
        "seed": seed,
        "partido": partido,
        "existe_solucion_ganadora": ganador == partido,
        "ganador": ganador,
        "fitness": fitness,
        "cand_vivo": cand_vivo,
        "rondas_sobrevive": rondas,
        "ronda_ganadora": ronda_ganadora,
        "vivos": json.dumps(vivos, ensure_ascii=False),
        "ultima_puntuacion": json.dumps(ultima_puntuacion, ensure_ascii=False),
        "puntuacion": json.dumps(puntuacion, ensure_ascii=False),
        "tiempo_segundos": duracion,
        "grafico_mejor_fitness": str(graficos["mejor_fitness_acumulado"]),
        "grafico_fitness_medio": str(graficos["fitness_medio_generacion"]),
    }


def guardar_resultados(resultados, output_path):
    # Guarda los resultados de las ejecuciones en CSV
    with output_path.open("w", encoding="utf-8", newline="") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=resultados[0].keys())
        writer.writeheader()
        writer.writerows(resultados)


def calcular_estadisticas_agregadas(resultados):
    # Resume las ejecuciones por partido
    resumenes = []

    for partido in dict.fromkeys(resultado["partido"] for resultado in resultados):
        ejecuciones = [
            resultado for resultado in resultados
            if resultado["partido"] == partido
        ]
        exitos = sum( resultado["existe_solucion_ganadora"] for resultado in ejecuciones )
        fitness = [resultado["fitness"] for resultado in ejecuciones]
        rondas = [resultado["rondas_sobrevive"] for resultado in ejecuciones]
        tiempos = [resultado["tiempo_segundos"] for resultado in ejecuciones]

        resumenes.append( { "partido": partido, "ejecuciones": len(ejecuciones), "soluciones_ganadoras": exitos, "existe_alguna_solucion_ganadora": exitos > 0, "porcentaje_exito": 100.0 * exitos / len(ejecuciones), "fitness_medio": mean(fitness), "fitness_mediano": median(fitness), "fitness_maximo": max(fitness), "rondas_medias": mean(rondas), "rondas_medianas": median(rondas), "tiempo_medio_segundos": mean(tiempos), "tiempo_mediano_segundos": median(tiempos), "tiempo_total_segundos": sum(tiempos), } )

    return resumenes


def mostrar_estadisticas_agregadas(resumenes):
    # Muestra las estadísticas agregadas por pantalla
    print("\n" + "=" * 72)
    print("ESTADÍSTICAS AGREGADAS POR PARTIDO")
    print("=" * 72)
    for resumen in resumenes:
        print(f"\nPartido: {resumen['partido']}")
        print( f"  Soluciones ganadoras: {resumen['soluciones_ganadoras']}/" f"{resumen['ejecuciones']} " f"({resumen['porcentaje_exito']:.1f} %)" )
        print( "  Existe alguna solución ganadora: " f"{'sí' if resumen['existe_alguna_solucion_ganadora'] else 'no'}" )
        print( f"  Fitness: media={resumen['fitness_medio']:.3f}, " f"mediana={resumen['fitness_mediano']:.3f}, " f"máximo={resumen['fitness_maximo']:.3f}" )
        print( f"  Rondas sobrevividas: media={resumen['rondas_medias']:.2f}, " f"mediana={resumen['rondas_medianas']:.2f}" )
        print( f"  Tiempo: media={resumen['tiempo_medio_segundos']:.2f} s, " f"mediana={resumen['tiempo_mediano_segundos']:.2f} s, " f"total={resumen['tiempo_total_segundos']:.2f} s" )


def parse_args():
    # Lee los argumentos de la línea de comandos
    parser = argparse.ArgumentParser( description=( "Busca una regla de puntuación que haga ganador a un partido usando " "P8 del CIS 3558 sobre el perfil reducido a los partidos conservados." ) )
    parser.add_argument( "--partidos", nargs="+", type=resolver_partido, default=PARTIDOS_OBJETIVO, help=( "Uno o varios partidos objetivo separados por espacios " f"(por defecto: {', '.join(PARTIDOS_OBJETIVO)})." ), )
    parser.add_argument( "--ejecuciones", type=int, default=CONFIGURACION_GENETICO["ejecuciones"] )
    parser.add_argument( "--generations", type=int, default=CONFIGURACION_GENETICO["generations"] )
    parser.add_argument( "--pop-size", type=int, default=CONFIGURACION_GENETICO["pop_size"] )
    parser.add_argument( "--mutation-prob", type=float, default=CONFIGURACION_GENETICO["mutation_prob"] )
    parser.add_argument("--k", type=int, default=CONFIGURACION_GENETICO["k"])
    parser.add_argument( "--selection-method", choices=("rank", "roulette"), default=CONFIGURACION_GENETICO["selection_method"], )
    parser.add_argument("--seed", type=int, default=CONFIGURACION_GENETICO["seed"])
    parser.add_argument("--a", type=int, default=CONFIGURACION_GENETICO["a"])
    parser.add_argument("--b", type=int, default=CONFIGURACION_GENETICO["b"])
    parser.add_argument("--csv-datos", type=Path, default=CSV_DATOS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument( "--solo-rankings", action="store_true", help="Genera el CSV de rankings sin ejecutar el algoritmo genético.", )
    args = parser.parse_args()
    if len(set(args.partidos)) != len(args.partidos):
        parser.error("La lista de partidos objetivo contiene partidos repetidos")
    if args.ejecuciones < 1 or args.generations < 1 or args.pop_size < 2:
        parser.error("ejecuciones y generations deben ser >= 1; pop-size debe ser >= 2")
    if args.k is not None and args.k < 0:
        parser.error("k debe ser >= 0")
    if args.a > args.b:
        parser.error("a debe ser menor o igual que b")
    return args


def main():
    # Ejecuta el experimento completo
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    participantes, perfil = cargar_rankings(args.csv_datos)
    rankings_path = args.output_dir / "rankings_pregunta_8.csv"
    guardar_rankings(participantes, rankings_path)
    con_respuesta = sum(bool(p["ranking"]) for p in participantes)
    print( f"Rankings guardados en {rankings_path} " f"({con_respuesta}/{len(participantes)} participantes con respuesta válida " f"tras reducir a los partidos conservados)." )

    if args.solo_rankings:
        return

    resultados = []
    for partido_idx, partido in enumerate(args.partidos):
        print(f"\nPartido objetivo: {partido}")
        for ejecucion in range(1, args.ejecuciones + 1):
            resultado = ejecutar_genetico( perfil=perfil, partido=partido, args=args, ejecucion=ejecucion, partido_idx=partido_idx, )
            resultados.append(resultado)
            print( f"Ejecución {ejecucion}/{args.ejecuciones}: " f"solución ganadora=" f"{'sí' if resultado['existe_solucion_ganadora'] else 'no'}, " f"ganador={resultado['ganador']}, " f"fitness={resultado['fitness']:.3f}" )

    resultados_path = args.output_dir / "resultados_ejecuciones.csv"
    guardar_resultados(resultados, resultados_path)
    print(f"Resultados guardados en {resultados_path}")

    resumenes = calcular_estadisticas_agregadas(resultados)
    resumen_path = args.output_dir / "resultados_agregados_por_partido.csv"
    guardar_resultados(resumenes, resumen_path)
    mostrar_estadisticas_agregadas(resumenes)
    print(f"\nEstadísticas agregadas guardadas en {resumen_path}")


if __name__ == "__main__":
    main()
