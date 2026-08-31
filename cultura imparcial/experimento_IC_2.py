from collections import Counter
import csv
import json
import math
from pathlib import Path
from random import Random
import random
from statistics import median
import time
import sys
from types import SimpleNamespace

SRC_DIR = Path(__file__).resolve().parent
MODULOS_DIR = SRC_DIR.parent
if str(MODULOS_DIR) not in sys.path:
    sys.path.insert(0, str(MODULOS_DIR))
from algoritmos_genéticos.alg_gen import genetic_election
from algoritmos_genéticos.graficos_alg_gen import crear_graficos_convergencia
from heatmap_cultura_imparcial import crear_heatmap_latex

# Configuración general

CANDIDATOS = list("ABCDEFGHIJ")
NUM_INSTANCIAS = 1
VOTANTES_POR_INSTANCIA = 10000
EJECUCIONES_POR_CASO = 10
SEED_BASE = 20260726


CONFIGURACION_GENETICO = {
    "nombre": "reducida",
    "generations": 150,
    "mutation_prob": 0.10,
    "pop_size": 50,
    "k": None,  # por defecto: número de candidatos - 1
}

PARAMETROS_FIJOS = {
    "a": 0,
    "b": 100,
}

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "ejemplos"
    / "experimento_IC_posicion_media"
)



def generar_ranking_ic(candidatos, num_votantes, seed):
    # Genera rankings estrictos aleatorios y agrupa los repetidos
    rng = Random(seed)
    conteo = Counter()

    for _ in range(num_votantes):
        orden = tuple(rng.sample(candidatos, len(candidatos)))
        conteo[orden] += 1

    return [(votos, [[candidato] for candidato in orden]) for orden, votos in conteo.most_common()]


def matriz_posiciones(ranking, candidatos):
    # Cuenta los votos de cada candidato en cada posición
    matriz = {
        candidato: [0 for _ in range(len(candidatos))]
        for candidato in candidatos
    }

    for votos, preferencias in ranking:
        for posicion, nivel in enumerate(preferencias):
            for candidato in nivel:
                matriz[candidato][posicion] += votos

    total = sum(sum(frecuencias) for frecuencias in matriz.values())
    if total == 0:
        raise ValueError("La instancia no contiene votantes.")

    return matriz


def calcular_estadisticas_posicion(matriz, candidatos, num_votantes):
    # Calcula la posición media y su desviación estandarizada bajo IC
    m = len(candidatos)
    esperanza_ic = (m + 1) / 2
    desviacion_tipica_ic = math.sqrt((m**2 - 1) / (12 * num_votantes))

    estadisticas = {}

    for candidato in candidatos:
        frecuencias = matriz[candidato]
        posicion_media = sum((posicion + 1) * frecuencia for posicion, frecuencia in enumerate(frecuencias)) / num_votantes

        estadisticas[candidato] = {
            "posicion_media": posicion_media,
            "esperanza_ic": esperanza_ic,
            "desviacion_tipica_ic": desviacion_tipica_ic,
            "desviacion_favorable": esperanza_ic - posicion_media,
            "z_ic": (
                (esperanza_ic - posicion_media) / desviacion_tipica_ic
                if desviacion_tipica_ic > 0
                else 0.0
            ),
        }

    # Rango 1 = mejor posición media; rango m = peor posición media.
    ordenados = sorted(candidatos, key=lambda c: (estadisticas[c]["posicion_media"], candidatos.index(c),),)

    for rango, candidato in enumerate(ordenados, start=1):
        estadisticas[candidato]["rango_posicion_media"] = rango

    return estadisticas


def seleccionar_candidatos_objetivo(estadisticas, candidatos):
    # Selecciona los candidatos mejor, neutral y peor según su posición media
    if len(candidatos) < 3:
        raise ValueError("Se necesitan al menos tres candidatos.")

    mejor = min(candidatos, key=lambda c: (estadisticas[c]["posicion_media"], candidatos.index(c),),)

    peor = max(candidatos, key=lambda c: (estadisticas[c]["posicion_media"], -candidatos.index(c),),)

    candidatos_neutrales = [
        candidato
        for candidato in candidatos
        if candidato not in {mejor, peor}
    ]

    neutral = min(candidatos_neutrales, key=lambda c: (abs(estadisticas[c]["posicion_media"] - estadisticas[c]["esperanza_ic"]), candidatos.index(c),),)

    seleccion = [
        ("mejor", mejor),
        ("neutral", neutral),
        ("peor", peor),
    ]

    objetivos = []
    for categoria, candidato in seleccion:
        datos = estadisticas[candidato]
        objetivos.append({"candidato": candidato, "categoria": categoria, "posicion_media": datos["posicion_media"], "esperanza_ic": datos["esperanza_ic"], "desviacion_favorable": datos["desviacion_favorable"], "z_ic": datos["z_ic"], "rango_posicion_media": datos["rango_posicion_media"],})

    return objetivos



def generar_heatmap(matriz, candidatos, estadisticas, instancia_id, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"heatmap_instancia_{instancia_id:02d}_2"
    posiciones_medias = {
        candidato: estadisticas[candidato]["posicion_media"]
        for candidato in candidatos
    }

    return crear_heatmap_latex(matriz=matriz, candidatos=candidatos, output_base=base, posiciones_medias=posiciones_medias, compilar_pdf=False,)


def parametros_genetico(configuracion):
    parametros = PARAMETROS_FIJOS.copy()
    parametros.update({"generations": configuracion["generations"], "mutation_prob": configuracion["mutation_prob"], "pop_size": configuracion["pop_size"], "k": configuracion["k"],})
    return parametros


def ejecutar_una_vez(ranking, candidato_objetivo, parametros, seed_ejecucion, output_dir):
    random.seed(seed_ejecucion)
    inicio = time.perf_counter()

    best_candidate, best_fitness, resultado, history = genetic_election(ranking=ranking, cand=candidato_objetivo, return_history=True, **parametros,)

    tiempo = time.perf_counter() - inicio
    ganador, vivos, cand_vivo, rondas_sobrevive, ultima_puntuacion = resultado

    output_dir.mkdir(parents=True, exist_ok=True)
    crear_graficos_convergencia(history, output_dir)

    return {"puntuacion": best_candidate, "best_fitness": best_fitness, "tiempo": tiempo, "ganador": ganador, "cand_vivo": cand_vivo, "rondas_sobrevive": rondas_sobrevive, "vivos": vivos, "ultima_puntuacion": ultima_puntuacion, "exito": ganador == candidato_objetivo,}


def serializar_puntuacion(puntuacion):
    return json.dumps(puntuacion, ensure_ascii=False, separators=(",", ":"),)


def ejecutar_caso(ranking, instancia_id, total_instancias, num_votantes, configuracion, objetivo, objetivo_idx, ejecuciones, seed_instancia, output_dir):
    # Ejecuta varias veces el algoritmo genético y resume el caso
    candidato_objetivo = objetivo["candidato"]
    parametros = parametros_genetico(configuracion)
    resultados = []

    print()
    print("-" * 80, flush=True)
    print(f"Instancia {instancia_id}/{total_instancias} | " f"objetivo={candidato_objetivo} | " f"categoría={objetivo['categoria']} | " f"posición media={objetivo['posicion_media']:.4f} | " f"z_IC={objetivo['z_ic']:.3f}", flush=True,)
    print(f"Parámetros del AG: {parametros}", flush=True)

    for ejecucion_idx in range(ejecuciones):
        seed_ejecucion = (
            seed_instancia * 1_000_000
            + objetivo_idx * 10_000
            + ejecucion_idx
        )

        graficos_dir = output_dir / "graficos" / f"{objetivo['categoria']}_{candidato_objetivo}" / f"ejecucion_{ejecucion_idx + 1:02d}"
        resultado = ejecutar_una_vez(ranking=ranking, candidato_objetivo=candidato_objetivo, parametros=parametros, seed_ejecucion=seed_ejecucion, output_dir=graficos_dir,)
        resultados.append(resultado)

        print(f"  Ejecución {ejecucion_idx + 1:02d}/{ejecuciones}: " f"ganador={resultado['ganador']} | " f"éxito={'sí' if resultado['exito'] else 'no'} | " f"rondas={resultado['rondas_sobrevive']} | " f"fitness={resultado['best_fitness']:.3f} | " f"tiempo={resultado['tiempo']:.2f}s", flush=True,)

    exitos = sum(resultado["exito"] for resultado in resultados)
    mejor_resultado = max(resultados, key=lambda resultado: resultado["best_fitness"],)

    resumen = {
        "instancia": instancia_id,
        "C": len(CANDIDATOS),
        "v": num_votantes,
        "rankings_distintos": len(ranking),
        "configuracion": configuracion["nombre"],
        "generations": configuracion["generations"],
        "pop_size": configuracion["pop_size"],
        "mutation_prob": configuracion["mutation_prob"],
        "c_star": candidato_objetivo,
        "categoria": objetivo["categoria"],
        "rango_posicion_media": objetivo["rango_posicion_media"],
        "posicion_media": objetivo["posicion_media"],
        "esperanza_ic": objetivo["esperanza_ic"],
        "desviacion_favorable": objetivo["desviacion_favorable"],
        "z_ic": objetivo["z_ic"],
        "ejecuciones": ejecuciones,
        "exitos": exitos,
        "exito_porcentaje": 100.0 * exitos / ejecuciones,
        "mediana_fitness": median(resultado["best_fitness"] for resultado in resultados),
        "mediana_rondas_sobrevive": median(resultado["rondas_sobrevive"] for resultado in resultados),
        "mediana_tiempo": median(resultado["tiempo"] for resultado in resultados),
        "mejor_fitness": mejor_resultado["best_fitness"],
        "mejor_puntuacion": serializar_puntuacion(mejor_resultado["puntuacion"]),
    }

    detalles = []
    for ejecucion_idx, resultado in enumerate(resultados, start=1):
        detalles.append({"instancia": instancia_id, "configuracion": configuracion["nombre"], "c_star": candidato_objetivo, "categoria": objetivo["categoria"], "rango_posicion_media": objetivo["rango_posicion_media"], "posicion_media": objetivo["posicion_media"], "z_ic": objetivo["z_ic"], "ejecucion": ejecucion_idx, "exito": resultado["exito"], "ganador": resultado["ganador"], "cand_vivo": resultado["cand_vivo"], "rondas_sobrevive": resultado["rondas_sobrevive"], "best_fitness": resultado["best_fitness"], "puntuacion": serializar_puntuacion(resultado["puntuacion"]), "tiempo": resultado["tiempo"],})

    print("Resumen del caso: " f"éxito={resumen['exito_porcentaje']:.1f}% | " f"mediana fitness={resumen['mediana_fitness']:.1f} | " f"mediana rondas={resumen['mediana_rondas_sobrevive']:.1f} | " f"mediana tiempo={resumen['mediana_tiempo']:.2f}s", flush=True,)

    return resumen, detalles


# Output

def guardar_csv(registros, output_path, campos):
    with output_path.open("w", newline="", encoding="utf-8",) as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos,)
        writer.writeheader()
        writer.writerows(registros)


def escapar_latex(texto):
    reemplazos = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(reemplazos.get(caracter, caracter) for caracter in str(texto))


def crear_tabla_latex(resumenes):
    filas = []
    instancia_anterior = None

    for resumen in resumenes:
        if (
            instancia_anterior is not None
            and resumen["instancia"] != instancia_anterior
        ):
            filas.append(r"\addlinespace")

        filas.append(f"{resumen['instancia']} & " f"${resumen['c_star']}$ & " f"{escapar_latex(resumen['categoria'])} & " f"{resumen['posicion_media']:.3f} & " f"{resumen['z_ic']:.2f} & " f"{resumen['ejecuciones']} & " f"{resumen['exito_porcentaje']:.1f} & " f"{resumen['mediana_fitness']:.1f} & " f"{resumen['mejor_fitness']:.1f} & " f"{resumen['mediana_rondas_sobrevive']:.1f} & " f"{resumen['mediana_tiempo']:.2f} \\\\")

        instancia_anterior = resumen["instancia"]

    cuerpo = "\n".join(filas)

    return rf"""\begin{{longtable}}{{@{{}}c c l c c c c c c c c@{{}}}}
\caption{{Resultados del algoritmo genético sobre instancias de cultura imparcial. En cada instancia se seleccionan como objetivos el candidato con mejor posición media, uno neutral y el candidato con peor posición media.}}
\label{{tab:resultados_ic_posicion_media}} \\
\toprule
\textbf{{Inst.}}
& \textbf{{$c^\star$}}
& \textbf{{Categoría}}
& \textbf{{Pos. media}}
& \textbf{{$z_{{IC}}$}}
& \textbf{{Ejec.}}
& \textbf{{Éxito (\%)}}
& \textbf{{Mediana fitness}}
& \textbf{{Mejor fitness}}
& \textbf{{Mediana rondas}}
& \textbf{{Mediana tiempo (s)}} \\
\midrule
\endfirsthead

\toprule
\textbf{{Inst.}}
& \textbf{{$c^\star$}}
& \textbf{{Categoría}}
& \textbf{{Pos. media}}
& \textbf{{$z_{{IC}}$}}
& \textbf{{Ejec.}}
& \textbf{{Éxito (\%)}}
& \textbf{{Mediana fitness}}
& \textbf{{Mejor fitness}}
& \textbf{{Mediana rondas}}
& \textbf{{Mediana tiempo (s)}} \\
\midrule
\endhead

{cuerpo}
\bottomrule
\end{{longtable}}
"""


def crear_resumen_categorias(detalles):
    # Resume las ejecuciones por categoría de candidato
    categorias = ["mejor", "neutral", "peor"]
    resumen = []

    for categoria in categorias:
        grupo = [
            registro
            for registro in detalles
            if registro["categoria"] == categoria
        ]

        if not grupo:
            continue

        exitos = sum(bool(registro["exito"]) for registro in grupo)

        resumen.append({"categoria": categoria, "ejecuciones": len(grupo), "exitos": exitos, "exito_porcentaje": 100.0 * exitos / len(grupo), "mediana_posicion_media": median(registro["posicion_media"] for registro in grupo), "mediana_z_ic": median(registro["z_ic"] for registro in grupo), "mediana_fitness": median(registro["best_fitness"] for registro in grupo), "mediana_rondas_sobrevive": median(registro["rondas_sobrevive"] for registro in grupo), "mediana_tiempo": median(registro["tiempo"] for registro in grupo),})

    return resumen


def guardar_resultados(resumenes, detalles, posiciones, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    resumen_path = output_dir / "resultados_resumen_2.csv"
    detalles_path = output_dir / "resultados_ejecuciones_2.csv"
    posiciones_path = output_dir / "posiciones_medias_2.csv"
    categorias_path = output_dir / "resumen_categorias_2.csv"
    tabla_path = output_dir / "tabla_resultados_2.tex"

    campos_resumen = [
        "instancia",
        "C",
        "v",
        "rankings_distintos",
        "configuracion",
        "generations",
        "pop_size",
        "mutation_prob",
        "c_star",
        "categoria",
        "rango_posicion_media",
        "posicion_media",
        "esperanza_ic",
        "desviacion_favorable",
        "z_ic",
        "ejecuciones",
        "exitos",
        "exito_porcentaje",
        "mediana_fitness",
        "mediana_rondas_sobrevive",
        "mediana_tiempo",
        "mejor_fitness",
        "mejor_puntuacion",
    ]

    campos_detalles = [
        "instancia",
        "configuracion",
        "c_star",
        "categoria",
        "rango_posicion_media",
        "posicion_media",
        "z_ic",
        "ejecucion",
        "exito",
        "ganador",
        "cand_vivo",
        "rondas_sobrevive",
        "best_fitness",
        "puntuacion",
        "tiempo",
    ]

    campos_posiciones = [
        "instancia",
        "seed_instancia",
        "candidato",
        "posicion_media",
        "rango_posicion_media",
        "esperanza_ic",
        "desviacion_favorable",
        "z_ic",
        "categoria_objetivo",
    ]

    resumen_categorias = crear_resumen_categorias(detalles)
    campos_categorias = [
        "categoria",
        "ejecuciones",
        "exitos",
        "exito_porcentaje",
        "mediana_posicion_media",
        "mediana_z_ic",
        "mediana_fitness",
        "mediana_rondas_sobrevive",
        "mediana_tiempo",
    ]

    guardar_csv(resumenes, resumen_path, campos_resumen,)
    guardar_csv(detalles, detalles_path, campos_detalles,)
    guardar_csv(posiciones, posiciones_path, campos_posiciones,)
    guardar_csv(resumen_categorias, categorias_path, campos_categorias,)

    tabla_path.write_text(crear_tabla_latex(resumenes), encoding="utf-8",)

    return (resumen_path, detalles_path, posiciones_path, categorias_path, tabla_path,)


def main():
    args = SimpleNamespace(
        votantes=VOTANTES_POR_INSTANCIA,
        instancias=NUM_INSTANCIAS,
        ejecuciones=EJECUCIONES_POR_CASO,
        generations=CONFIGURACION_GENETICO["generations"],
        pop_size=CONFIGURACION_GENETICO["pop_size"],
        mutation_prob=CONFIGURACION_GENETICO["mutation_prob"],
        k=CONFIGURACION_GENETICO["k"],
    )

    if args.votantes <= 0:
        raise ValueError("El número de votantes debe ser positivo.")
    if args.instancias <= 0:
        raise ValueError("El número de instancias debe ser positivo.")
    if args.ejecuciones <= 0:
        raise ValueError("El número de ejecuciones debe ser positivo.")
    if args.generations <= 0:
        raise ValueError("El número de generaciones debe ser positivo.")
    if args.pop_size <= 0:
        raise ValueError("El tamaño de población debe ser positivo.")
    if not 0 <= args.mutation_prob <= 1:
        raise ValueError("La probabilidad de mutación debe pertenecer a [0,1].")
    if args.k is not None and args.k < 0:
        raise ValueError("k debe ser mayor o igual que 0.")

    configuracion = {
        "nombre": "fija",
        "generations": args.generations,
        "mutation_prob": args.mutation_prob,
        "pop_size": args.pop_size,
        "k": args.k,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    resumenes = []
    detalles = []
    posiciones = []

    total_casos = args.instancias * 3
    caso_actual = 0

    print("Inicio del experimento IC según posición media", flush=True,)
    print(f"Salida: {OUTPUT_DIR}", flush=True)
    print(f"Candidatos: {CANDIDATOS}", flush=True)
    print(f"Instancias: {args.instancias}", flush=True)
    print(f"Votantes por instancia: {args.votantes}", flush=True)
    print(f"Ejecuciones por caso: {args.ejecuciones}", flush=True)
    print(f"Configuración AG: {configuracion}", flush=True)
    print(f"Número total de casos: {total_casos}", flush=True)
    print("Número total de ejecuciones del genético: " f"{total_casos * args.ejecuciones}", flush=True,)

    for instancia_idx in range(args.instancias):
        instancia_id = instancia_idx + 1
        seed_instancia = SEED_BASE + instancia_idx
        instancia_dir = OUTPUT_DIR / f"instancia_{instancia_id:02d}"
        resumenes_instancia = []
        detalles_instancia = []
        posiciones_instancia = []

        ranking = generar_ranking_ic(candidatos=CANDIDATOS, num_votantes=args.votantes, seed=seed_instancia,)

        matriz = matriz_posiciones(ranking, CANDIDATOS,)

        estadisticas = calcular_estadisticas_posicion(matriz=matriz, candidatos=CANDIDATOS, num_votantes=args.votantes,)

        objetivos = seleccionar_candidatos_objetivo(estadisticas=estadisticas, candidatos=CANDIDATOS,)

        categoria_por_candidato = {
            objetivo["candidato"]: objetivo["categoria"]
            for objetivo in objetivos
        }

        for candidato in CANDIDATOS:
            datos = estadisticas[candidato]
            registro_posicion = {"instancia": instancia_id, "seed_instancia": seed_instancia, "candidato": candidato, "posicion_media": datos["posicion_media"], "rango_posicion_media": datos["rango_posicion_media"], "esperanza_ic": datos["esperanza_ic"], "desviacion_favorable": datos["desviacion_favorable"], "z_ic": datos["z_ic"], "categoria_objetivo": (categoria_por_candidato.get(candidato, "")),}
            posiciones.append(registro_posicion)
            posiciones_instancia.append(registro_posicion)

        print()
        print("=" * 80, flush=True)
        print(f"Instancia {instancia_id}/{args.instancias}: " f"seed={seed_instancia}, " f"rankings distintos={len(ranking)}", flush=True,)

        print("Objetivos seleccionados:", flush=True)
        for objetivo in objetivos:
            print(f"  {objetivo['categoria']:>7}: " f"{objetivo['candidato']} | " f"posición media={objetivo['posicion_media']:.4f} | " f"z_IC={objetivo['z_ic']:.3f}", flush=True,)

        _pdf_path, svg_path, tex_path = generar_heatmap(matriz=matriz, candidatos=CANDIDATOS, estadisticas=estadisticas, instancia_id=instancia_id, output_dir=instancia_dir,)

        print(f"Heatmap SVG: {svg_path}", flush=True)
        print(f"Heatmap TikZ: {tex_path}", flush=True)

        for objetivo_idx, objetivo in enumerate(objetivos):
            caso_actual += 1
            print(f"Caso {caso_actual}/{total_casos}", flush=True,)

            resumen, detalles_caso = ejecutar_caso(ranking=ranking, instancia_id=instancia_id, total_instancias=args.instancias, num_votantes=args.votantes, configuracion=configuracion, objetivo=objetivo, objetivo_idx=objetivo_idx, ejecuciones=args.ejecuciones, seed_instancia=seed_instancia, output_dir=instancia_dir,)

            resumenes.append(resumen)
            detalles.extend(detalles_caso)
            resumenes_instancia.append(resumen)
            detalles_instancia.extend(detalles_caso)

            # Guardado parcial por seguridad.
            guardar_resultados(resumenes=resumenes, detalles=detalles, posiciones=posiciones, output_dir=OUTPUT_DIR,)
            guardar_resultados(resumenes=resumenes_instancia, detalles=detalles_instancia, posiciones=posiciones_instancia, output_dir=instancia_dir,)

    (
        resumen_path,
        detalles_path,
        posiciones_path,
        categorias_path,
        tabla_path,
    ) = guardar_resultados(resumenes=resumenes, detalles=detalles, posiciones=posiciones, output_dir=OUTPUT_DIR,)

    print()
    print("=" * 80, flush=True)
    print("Experimento terminado.", flush=True)
    print(f"Resumen por casos: {resumen_path}", flush=True)
    print(f"Ejecuciones detalladas: {detalles_path}", flush=True)
    print(f"Posiciones medias: {posiciones_path}", flush=True)
    print(f"Resumen por categorías: {categorias_path}", flush=True)
    print(f"Tabla LaTeX: {tabla_path}", flush=True)


if __name__ == "__main__":
    main()
