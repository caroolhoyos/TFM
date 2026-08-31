from collections import Counter, defaultdict
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

CULTURA_IMPARCIAL_DIR = MODULOS_DIR / "cultura imparcial"
if str(CULTURA_IMPARCIAL_DIR) not in sys.path:
    sys.path.insert(0, str(CULTURA_IMPARCIAL_DIR))

from algoritmos_genéticos.alg_gen import genetic_election
from algoritmos_genéticos.graficos_alg_gen import crear_graficos_convergencia
from heatmap_cultura_imparcial import crear_heatmap_latex

# Configuración general

CANDIDATOS = list("ABCDE")
VOTANTES_POR_INSTANCIA = 10000
INSTANCIAS_POR_CASO = 30
EJECUCIONES_POR_OBJETIVO = 5
SEED_BASE = 20260729

CONFIGURACION_GENETICO = {
    "nombre": "exploratoria",
    "generations": 150,
    "mutation_prob": 0.10,
    "pop_size": 50,
    "k": None,  # por defecto: número de candidatos - 1
}

PARAMETROS_FIJOS = {
    "a": 0,
    "b": 5,
}

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "resultados"
    / "experimento_mallows_1"
)

# Evalúa todos los candidatos para comparar sus resultados en cada instancia
OBJETIVOS_BASE = [
    (f"polo{indice}", candidato)
    for indice, candidato in enumerate(CANDIDATOS, start=1)
]


# Casos de un único centro


CASOS_UN_CENTRO = [
    {
        "nombre": "un_centro_muy_concentrado",
        "descripcion": "Un solo centro y preferencias muy concentradas.",
        "bloques": [
            {"centro": "ABCDE", "phi": 0.10, "proporcion": 1.0},
        ],
        "objetivos": OBJETIVOS_BASE,
    },
    {
        "nombre": "un_centro_concentracion_media",
        "descripcion": "Un solo centro con dispersión intermedia.",
        "bloques": [
            {"centro": "ABCDE", "phi": 0.50, "proporcion": 1.0},
        ],
        "objetivos": OBJETIVOS_BASE,
    },
    {
        "nombre": "un_centro_muy_disperso",
        "descripcion": "Un solo centro, pero rankings bastante dispersos.",
        "bloques": [
            {"centro": "ABCDE", "phi": 0.85, "proporcion": 1.0},
        ],
        "objetivos": OBJETIVOS_BASE,
    },

]


# Casos con dos centros

CASOS_DOS_CENTROS = [
    {
        "nombre": "dos_polos_equilibrados",
        "descripcion": (
            "Dos bloques igual de grandes, muy cohesionados y con centros "
            "opuestos."
        ),
        "bloques": [
            {"centro": "ABCDE", "phi": 0.15, "proporcion": 0.50},
            {"centro": "EDCBA", "phi": 0.15, "proporcion": 0.50},
        ],
        "objetivos": OBJETIVOS_BASE,
    },
    {
        "nombre": "dos_polos_mayoria_60_40",
        "descripcion": (
            "Los centros son opuestos, pero el primer bloque tiene mayoría."
        ),
        "bloques": [
            {"centro": "ABCDE", "phi": 0.15, "proporcion": 0.65},
            {"centro": "EDCBA", "phi": 0.15, "proporcion": 0.35},
        ],
        "objetivos": OBJETIVOS_BASE,
    },
    {
        "nombre": "minoria_pequena_pero_cohesionada",
        "descripcion": (
            "Una mayoría dispersa compite con una minoría muy cohesionada "
            "alrededor del centro opuesto."
        ),
        "bloques": [
            {"centro": "ABCDE", "phi": 0.85, "proporcion": 0.70},
            {"centro": "EDCBA", "phi": 0.10, "proporcion": 0.30},
        ],
        "objetivos": OBJETIVOS_BASE,
    },
]


# Generación Mallows con distancia de Kendall


def validar_centro(centro, candidatos):
    # Comprueba que el centro contiene todos los candidatos
    if isinstance(centro, str):
        centro = list(centro)
    else:
        centro = list(centro)

    if len(centro) != len(candidatos) or set(centro) != set(candidatos):
        raise ValueError(f"El centro {centro} no es una permutación de {candidatos}.")

    return centro


def muestrear_mallows_kendall(centro, phi, rng):
    # Genera un ranking con el modelo Mallows-Kendall
    if not 0.0 <= phi <= 1.0:
        raise ValueError("phi debe pertenecer al intervalo [0,1].")

    ranking = []

    for i, candidato in enumerate(centro):
        if phi == 0.0:
            posicion = i
        else:
            posiciones = list(range(i + 1))
            pesos = [phi ** (i - j) for j in posiciones]
            posicion = rng.choices(posiciones, weights=pesos, k=1)[0]

        ranking.insert(posicion, candidato)

    return tuple(ranking)


def repartir_votantes(num_votantes, proporciones):
    # Reparte los votantes por bloques sin perderlos por redondeo
    if num_votantes <= 0:
        raise ValueError("El número de votantes debe ser positivo.")
    if not proporciones:
        raise ValueError("Debe existir al menos un bloque de votantes.")
    if any(p < 0 for p in proporciones):
        raise ValueError("Las proporciones no pueden ser negativas.")

    suma = sum(proporciones)
    if suma <= 0:
        raise ValueError("La suma de las proporciones debe ser positiva.")

    normalizadas = [p / suma for p in proporciones]
    cantidades_reales = [num_votantes * p for p in normalizadas]
    cantidades = [math.floor(x) for x in cantidades_reales]

    restantes = num_votantes - sum(cantidades)
    orden_restos = sorted(range(len(cantidades)), key=lambda i: cantidades_reales[i] - cantidades[i], reverse=True,)

    for i in orden_restos[:restantes]:
        cantidades[i] += 1

    return cantidades


def generar_ranking_mallows_mixto(candidatos, bloques, num_votantes, seed):
    # Genera y agrupa rankings de uno o varios bloques Mallows
    rng = Random(seed)

    centros = [
        validar_centro(bloque["centro"], candidatos)
        for bloque in bloques
    ]
    proporciones = [bloque["proporcion"] for bloque in bloques]
    cantidades = repartir_votantes(num_votantes, proporciones)

    conteo = Counter()

    for bloque_idx, (bloque, centro, cantidad) in enumerate(zip(bloques, centros, cantidades)):
        phi = float(bloque["phi"])

        # Usa una semilla independiente y reproducible para cada bloque
        seed_bloque = rng.randrange(0, 2**63) + bloque_idx
        rng_bloque = Random(seed_bloque)

        for _ in range(cantidad):
            orden = muestrear_mallows_kendall(centro=centro, phi=phi, rng=rng_bloque,)
            conteo[orden] += 1

    ranking = [
        (votos, [[candidato] for candidato in orden])
        for orden, votos in conteo.most_common()
    ]

    metadatos_bloques = []
    for bloque, cantidad in zip(bloques, cantidades):
        metadatos_bloques.append({"centro": "".join(bloque["centro"]), "phi": float(bloque["phi"]), "proporcion": float(bloque["proporcion"]), "votantes": cantidad,})

    return ranking, metadatos_bloques


# Estadísticas descriptivas


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

    return matriz


def calcular_estadisticas_posicion(matriz, candidatos, num_votantes):
    # Calcula las estadísticas de posición de cada candidato
    estadisticas = {}

    for candidato in candidatos:
        frecuencias = matriz[candidato]
        total = sum(frecuencias)

        if total != num_votantes:
            raise ValueError(f"El candidato {candidato} aparece {total} veces, " f"pero deberían ser {num_votantes}.")

        posicion_media = sum((posicion + 1) * frecuencia for posicion, frecuencia in enumerate(frecuencias)) / num_votantes

        varianza_posicion = sum(((posicion + 1) - posicion_media) ** 2 * frecuencia for posicion, frecuencia in enumerate(frecuencias)) / num_votantes

        estadisticas[candidato] = {
            "posicion_media": posicion_media,
            "desviacion_posicion": math.sqrt(varianza_posicion),
            "frecuencia_primera": frecuencias[0] / num_votantes,
            "frecuencia_ultima": frecuencias[-1] / num_votantes,
        }

    ordenados = sorted(candidatos, key=lambda c: (estadisticas[c]["posicion_media"], candidatos.index(c),),)

    for rango, candidato in enumerate(ordenados, start=1):
        estadisticas[candidato]["rango_posicion_media"] = rango

    return estadisticas


def posiciones_en_centros(candidato, bloques):
    # Obtiene la posición del candidato en cada centro
    posiciones = []

    for bloque in bloques:
        centro = list(bloque["centro"])
        posiciones.append(centro.index(candidato) + 1)

    return posiciones


def generar_heatmap(matriz, candidatos, estadisticas, bloque_experimental, caso_nombre, instancia_id, output_dir):
    # Genera el heatmap de una instancia
    carpeta = (
        output_dir
        / bloque_experimental
        / caso_nombre
        / f"instancia_{instancia_id:02d}"
    )
    carpeta.mkdir(parents=True, exist_ok=True)

    base = carpeta / f"heatmap_instancia_{instancia_id:02d}"
    posiciones_medias = {
        candidato: estadisticas[candidato]["posicion_media"]
        for candidato in candidatos
    }

    return crear_heatmap_latex(matriz=matriz, candidatos=candidatos, output_base=base, posiciones_medias=posiciones_medias, compilar_pdf=False,)


# Ejecución del AG


def parametros_genetico(configuracion):
    # Combina la configuración genética con los parámetros fijos
    parametros = PARAMETROS_FIJOS.copy()
    parametros.update({"generations": configuracion["generations"], "mutation_prob": configuracion["mutation_prob"], "pop_size": configuracion["pop_size"], "k": configuracion["k"],})
    return parametros


def ejecutar_una_vez(ranking, candidato_objetivo, parametros, seed_ejecucion, output_dir):
    # Ejecuta una vez el algoritmo genético y guarda sus gráficos
    random.seed(seed_ejecucion)
    inicio = time.perf_counter()

    best_candidate, best_fitness, resultado, history = genetic_election(ranking=ranking, cand=candidato_objetivo, return_history=True, **parametros,)

    tiempo = time.perf_counter() - inicio
    ganador, vivos, cand_vivo, rondas_sobrevive, ultima_puntuacion = resultado

    output_dir.mkdir(parents=True, exist_ok=True)
    crear_graficos_convergencia(history, output_dir)

    return {"puntuacion": best_candidate, "best_fitness": best_fitness, "tiempo": tiempo, "ganador": ganador, "cand_vivo": cand_vivo, "rondas_sobrevive": rondas_sobrevive, "vivos": vivos, "ultima_puntuacion": ultima_puntuacion, "exito": ganador == candidato_objetivo,}


def serializar(valor):
    # Convierte un valor a JSON compacto
    return json.dumps(valor, ensure_ascii=False, separators=(",", ":"),)


def ejecutar_objetivo(ranking, bloque_experimental, caso, instancia_id, total_instancias_caso, num_votantes, configuracion, categoria_objetivo, candidato_objetivo, objetivo_idx, ejecuciones, seed_instancia, estadisticas, metadatos_bloques, output_dir):
    # Ejecuta y resume las pruebas de un candidato objetivo
    parametros = parametros_genetico(configuracion)
    resultados = []
    datos_candidato = estadisticas[candidato_objetivo]
    posiciones_centrales = posiciones_en_centros(candidato_objetivo, caso["bloques"],)

    print()
    print("-" * 88, flush=True)
    print(f"{bloque_experimental} | caso={caso['nombre']} | " f"instancia={instancia_id}/{total_instancias_caso} | " f"objetivo={candidato_objetivo} ({categoria_objetivo})", flush=True,)
    print(f"posición media={datos_candidato['posicion_media']:.3f} | " f"desv. posición={datos_candidato['desviacion_posicion']:.3f} | " f"freq. primero={datos_candidato['frecuencia_primera']:.3f}", flush=True,)

    for ejecucion_idx in range(ejecuciones):
        seed_ejecucion = (
            seed_instancia * 100_000
            + objetivo_idx * 1_000
            + ejecucion_idx
        )

        graficos_dir = (
            output_dir
            / "graficos"
            / f"{categoria_objetivo}_{candidato_objetivo}"
            / f"ejecucion_{ejecucion_idx + 1:02d}"
        )
        resultado = ejecutar_una_vez(ranking=ranking, candidato_objetivo=candidato_objetivo, parametros=parametros, seed_ejecucion=seed_ejecucion, output_dir=graficos_dir,)
        resultados.append(resultado)

        print(f"  Ejecución {ejecucion_idx + 1:02d}/{ejecuciones}: " f"ganador={resultado['ganador']} | " f"éxito={'sí' if resultado['exito'] else 'no'} | " f"rondas={resultado['rondas_sobrevive']} | " f"fitness={resultado['best_fitness']:.3f} | " f"tiempo={resultado['tiempo']:.2f}s", flush=True,)

    exitos = sum(resultado["exito"] for resultado in resultados)
    mejor_resultado = max(resultados, key=lambda resultado: resultado["best_fitness"],)

    campos_comunes = {
        "bloque_experimental": bloque_experimental,
        "caso": caso["nombre"],
        "descripcion_caso": caso["descripcion"],
        "instancia": instancia_id,
        "C": len(CANDIDATOS),
        "v": num_votantes,
        "rankings_distintos": len(ranking),
        "bloques_mallows": serializar(metadatos_bloques),
        "configuracion": configuracion["nombre"],
        "generations": configuracion["generations"],
        "pop_size": configuracion["pop_size"],
        "mutation_prob": configuracion["mutation_prob"],
        "c_star": candidato_objetivo,
        "categoria_objetivo": categoria_objetivo,
        "posiciones_en_centros": serializar(posiciones_centrales),
        "rango_posicion_media": datos_candidato["rango_posicion_media"],
        "posicion_media": datos_candidato["posicion_media"],
        "desviacion_posicion": datos_candidato["desviacion_posicion"],
        "frecuencia_primera": datos_candidato["frecuencia_primera"],
        "frecuencia_ultima": datos_candidato["frecuencia_ultima"],
    }

    resumen = {
        **campos_comunes,
        "ejecuciones": ejecuciones,
        "exitos": exitos,
        "exito_porcentaje": 100.0 * exitos / ejecuciones,
        # Guarda indicadores para comparar las instancias
        "alguna_solucion_encontrada": exitos > 0,
        "todas_ejecuciones_exitosas": exitos == ejecuciones,
        "mediana_fitness": median(resultado["best_fitness"] for resultado in resultados),
        "mediana_rondas_sobrevive": median(resultado["rondas_sobrevive"] for resultado in resultados),
        "mediana_tiempo": median(resultado["tiempo"] for resultado in resultados),
        "mejor_fitness": mejor_resultado["best_fitness"],
        "mejor_puntuacion": serializar(mejor_resultado["puntuacion"]),
    }

    detalles = []
    for ejecucion_idx, resultado in enumerate(resultados, start=1):
        detalles.append({**campos_comunes, "ejecucion": ejecucion_idx, "exito": resultado["exito"], "ganador": resultado["ganador"], "cand_vivo": resultado["cand_vivo"], "rondas_sobrevive": resultado["rondas_sobrevive"], "best_fitness": resultado["best_fitness"], "puntuacion": serializar(resultado["puntuacion"]), "tiempo": resultado["tiempo"],})

    print("Resumen: " f"éxito={resumen['exito_porcentaje']:.1f}% | " f"mediana fitness={resumen['mediana_fitness']:.3f} | " f"mediana rondas={resumen['mediana_rondas_sobrevive']:.1f}", flush=True,)

    return resumen, detalles


# Salida de resultados


def guardar_csv(registros, output_path, campos):
    # Guarda una colección de registros en CSV
    with output_path.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(registros)


def crear_resumen_global(detalles):
    # Agrupa las ejecuciones por caso y candidato objetivo
    grupos = defaultdict(list)

    for registro in detalles:
        clave = (
            registro["bloque_experimental"],
            registro["caso"],
            registro["c_star"],
            registro["categoria_objetivo"],
        )
        grupos[clave].append(registro)

    resumen = []

    for clave, grupo in grupos.items():
        bloque_experimental, caso, c_star, categoria = clave
        exitos = sum(bool(registro["exito"]) for registro in grupo)

        resumen.append({"bloque_experimental": bloque_experimental, "caso": caso, "c_star": c_star, "categoria_objetivo": categoria, "ejecuciones": len(grupo), "exitos": exitos, "exito_porcentaje": 100.0 * exitos / len(grupo), "mediana_posicion_media": median(registro["posicion_media"] for registro in grupo), "mediana_desviacion_posicion": median(registro["desviacion_posicion"] for registro in grupo), "mediana_frecuencia_primera": median(registro["frecuencia_primera"] for registro in grupo), "mediana_fitness": median(registro["best_fitness"] for registro in grupo), "mediana_rondas_sobrevive": median(registro["rondas_sobrevive"] for registro in grupo), "mediana_tiempo": median(registro["tiempo"] for registro in grupo),})

    return sorted(resumen, key=lambda r: (r["bloque_experimental"], r["caso"], CANDIDATOS.index(r["c_star"]),),)


def crear_resumen_entre_instancias(resumenes):
    # Resume los resultados usando cada instancia como unidad experimental
    grupos = defaultdict(list)

    for registro in resumenes:
        clave = (
            registro["bloque_experimental"],
            registro["caso"],
            registro["c_star"],
            registro["categoria_objetivo"],
        )
        grupos[clave].append(registro)

    resumen = []

    for clave, grupo in grupos.items():
        bloque_experimental, caso, c_star, categoria = clave

        porcentajes_por_instancia = [
            registro["exito_porcentaje"] for registro in grupo
        ]
        ejecuciones_totales = sum(registro["ejecuciones"] for registro in grupo)
        exitos_totales = sum(registro["exitos"] for registro in grupo)

        instancias_con_algun_exito = sum(registro["exitos"] > 0 for registro in grupo)
        instancias_con_exito_total = sum(registro["exitos"] == registro["ejecuciones"] for registro in grupo)

        num_instancias = len(grupo)

        resumen.append({"bloque_experimental": bloque_experimental, "caso": caso, "c_star": c_star, "categoria_objetivo": categoria, "num_instancias": num_instancias, "instancias_con_algun_exito": (instancias_con_algun_exito), "porcentaje_instancias_con_algun_exito": (100.0 * instancias_con_algun_exito / num_instancias), "instancias_con_exito_total": instancias_con_exito_total, "porcentaje_instancias_con_exito_total": (100.0 * instancias_con_exito_total / num_instancias), "ejecuciones_totales": ejecuciones_totales, "exitos_totales": exitos_totales, "exito_global_porcentaje": (100.0 * exitos_totales / ejecuciones_totales if ejecuciones_totales > 0 else 0.0), "media_exito_por_instancia": (sum(porcentajes_por_instancia) / num_instancias), "mediana_exito_por_instancia": median(porcentajes_por_instancia), "minimo_exito_por_instancia": min(porcentajes_por_instancia), "maximo_exito_por_instancia": max(porcentajes_por_instancia), "mediana_posicion_media_entre_instancias": median(registro["posicion_media"] for registro in grupo), "mediana_desviacion_posicion_entre_instancias": median(registro["desviacion_posicion"] for registro in grupo), "mediana_frecuencia_primera_entre_instancias": median(registro["frecuencia_primera"] for registro in grupo), "mediana_fitness_entre_instancias": median(registro["mediana_fitness"] for registro in grupo), "mediana_rondas_entre_instancias": median(registro["mediana_rondas_sobrevive"] for registro in grupo), "mediana_tiempo_entre_instancias": median(registro["mediana_tiempo"] for registro in grupo),})

    return sorted(resumen, key=lambda r: (r["bloque_experimental"], r["caso"], CANDIDATOS.index(r["c_star"]),),)


def escapar_latex(texto):
    # Escapa los caracteres especiales de LaTeX
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
    return "".join(reemplazos.get(c, c) for c in str(texto))


def crear_tabla_latex(resumen_global):
    # Genera la tabla LaTeX del resumen global
    filas = []

    for registro in resumen_global:
        filas.append(f"{escapar_latex(registro['caso'])} & " f"${registro['c_star']}$ & " f"{registro['mediana_posicion_media']:.2f} & " f"{registro['mediana_desviacion_posicion']:.2f} & " f"{100 * registro['mediana_frecuencia_primera']:.1f} & " f"{registro['ejecuciones']} & " f"{registro['exito_porcentaje']:.1f} & " f"{registro['mediana_fitness']:.2f} & " f"{registro['mediana_rondas_sobrevive']:.1f} \\\\")

    cuerpo = "\n".join(filas)

    return rf"""\begin{{longtable}}{{@{{}}l c c c c c c c c@{{}}}}
\caption{{Resultados agregados del algoritmo genético sobre los casos Mallows.}}
\label{{tab:resultados_mallows}} \\
\toprule
\textbf{{Caso}}
& \textbf{{$c^\star$}}
& \textbf{{Pos. media}}
& \textbf{{Desv. pos.}}
& \textbf{{Primero (\%)}}
& \textbf{{Ejec.}}
& \textbf{{Éxito (\%)}}
& \textbf{{Fitness}}
& \textbf{{Rondas}} \\
\midrule
\endfirsthead

\toprule
\textbf{{Caso}}
& \textbf{{$c^\star$}}
& \textbf{{Pos. media}}
& \textbf{{Desv. pos.}}
& \textbf{{Primero (\%)}}
& \textbf{{Ejec.}}
& \textbf{{Éxito (\%)}}
& \textbf{{Fitness}}
& \textbf{{Rondas}} \\
\midrule
\endhead

{cuerpo}
\bottomrule
\end{{longtable}}
"""


def crear_tabla_instancias_latex(resumen_instancias):
    # Genera la tabla LaTeX del resumen por instancias
    filas = []

    for registro in resumen_instancias:
        filas.append(f"{escapar_latex(registro['caso'])} & " f"${registro['c_star']}$ & " f"{registro['num_instancias']} & " f"{registro['instancias_con_algun_exito']} & " f"{registro['porcentaje_instancias_con_algun_exito']:.1f} & " f"{registro['mediana_exito_por_instancia']:.1f} & " f"{registro['minimo_exito_por_instancia']:.1f} & " f"{registro['maximo_exito_por_instancia']:.1f} & " f"{registro['exito_global_porcentaje']:.1f} \\\\")

    cuerpo = "\n".join(filas)

    return rf"""\begin{{longtable}}{{@{{}}l c c c c c c c c@{{}}}}
\caption{{Resultados del algoritmo genético agregados usando la instancia como unidad experimental.}}
\label{{tab:resultados_mallows_instancias}} \\
\toprule
\textbf{{Caso}}
& \textbf{{$c^\star$}}
& \textbf{{Inst.}}
& \textbf{{Inst. con éxito}}
& \textbf{{Inst. con éxito (\%)}}
& \textbf{{Mediana éxito/inst.}}
& \textbf{{Mín.}}
& \textbf{{Máx.}}
& \textbf{{Éxito ejec. (\%)}} \\
\midrule
\endfirsthead

\toprule
\textbf{{Caso}}
& \textbf{{$c^\star$}}
& \textbf{{Inst.}}
& \textbf{{Inst. con éxito}}
& \textbf{{Inst. con éxito (\%)}}
& \textbf{{Mediana éxito/inst.}}
& \textbf{{Mín.}}
& \textbf{{Máx.}}
& \textbf{{Éxito ejec. (\%)}} \\
\midrule
\endhead

{cuerpo}
\bottomrule
\end{{longtable}}
"""


def guardar_resultados(resumenes, detalles, posiciones, output_dir):
    # Guarda los resultados y sus resúmenes en varios formatos
    output_dir.mkdir(parents=True, exist_ok=True)

    resumen_path = output_dir / "resultados_por_instancia.csv"
    detalles_path = output_dir / "resultados_ejecuciones.csv"
    posiciones_path = output_dir / "estadisticas_candidatos.csv"
    global_path = output_dir / "resumen_global.csv"
    instancias_path = output_dir / "resumen_entre_instancias.csv"
    tabla_path = output_dir / "tabla_resultados.tex"
    tabla_instancias_path = output_dir / "tabla_resultados_instancias.tex"

    campos_resumen = [
        "bloque_experimental",
        "caso",
        "descripcion_caso",
        "instancia",
        "C",
        "v",
        "rankings_distintos",
        "bloques_mallows",
        "configuracion",
        "generations",
        "pop_size",
        "mutation_prob",
        "c_star",
        "categoria_objetivo",
        "posiciones_en_centros",
        "rango_posicion_media",
        "posicion_media",
        "desviacion_posicion",
        "frecuencia_primera",
        "frecuencia_ultima",
        "ejecuciones",
        "exitos",
        "exito_porcentaje",
        "alguna_solucion_encontrada",
        "todas_ejecuciones_exitosas",
        "mediana_fitness",
        "mediana_rondas_sobrevive",
        "mediana_tiempo",
        "mejor_fitness",
        "mejor_puntuacion",
    ]

    campos_detalles = [
        "bloque_experimental",
        "caso",
        "descripcion_caso",
        "instancia",
        "C",
        "v",
        "rankings_distintos",
        "bloques_mallows",
        "configuracion",
        "generations",
        "pop_size",
        "mutation_prob",
        "c_star",
        "categoria_objetivo",
        "posiciones_en_centros",
        "rango_posicion_media",
        "posicion_media",
        "desviacion_posicion",
        "frecuencia_primera",
        "frecuencia_ultima",
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
        "bloque_experimental",
        "caso",
        "descripcion_caso",
        "instancia",
        "seed_instancia",
        "bloques_mallows",
        "candidato",
        "posiciones_en_centros",
        "posicion_media",
        "desviacion_posicion",
        "frecuencia_primera",
        "frecuencia_ultima",
        "rango_posicion_media",
        "categoria_objetivo",
    ]

    resumen_global = crear_resumen_global(detalles)
    resumen_instancias = crear_resumen_entre_instancias(resumenes)

    campos_global = [
        "bloque_experimental",
        "caso",
        "c_star",
        "categoria_objetivo",
        "ejecuciones",
        "exitos",
        "exito_porcentaje",
        "mediana_posicion_media",
        "mediana_desviacion_posicion",
        "mediana_frecuencia_primera",
        "mediana_fitness",
        "mediana_rondas_sobrevive",
        "mediana_tiempo",
    ]

    campos_instancias = [
        "bloque_experimental",
        "caso",
        "c_star",
        "categoria_objetivo",
        "num_instancias",
        "instancias_con_algun_exito",
        "porcentaje_instancias_con_algun_exito",
        "instancias_con_exito_total",
        "porcentaje_instancias_con_exito_total",
        "ejecuciones_totales",
        "exitos_totales",
        "exito_global_porcentaje",
        "media_exito_por_instancia",
        "mediana_exito_por_instancia",
        "minimo_exito_por_instancia",
        "maximo_exito_por_instancia",
        "mediana_posicion_media_entre_instancias",
        "mediana_desviacion_posicion_entre_instancias",
        "mediana_frecuencia_primera_entre_instancias",
        "mediana_fitness_entre_instancias",
        "mediana_rondas_entre_instancias",
        "mediana_tiempo_entre_instancias",
    ]

    guardar_csv(resumenes, resumen_path, campos_resumen)
    guardar_csv(detalles, detalles_path, campos_detalles)
    guardar_csv(posiciones, posiciones_path, campos_posiciones)
    guardar_csv(resumen_global, global_path, campos_global)
    guardar_csv(resumen_instancias, instancias_path, campos_instancias)

    tabla_path.write_text(crear_tabla_latex(resumen_global), encoding="utf-8",)
    tabla_instancias_path.write_text(crear_tabla_instancias_latex(resumen_instancias), encoding="utf-8",)

    return (resumen_path, detalles_path, posiciones_path, global_path, instancias_path, tabla_path, tabla_instancias_path,)


# CLI y programa principal


def seleccionar_casos(bloque, nombres_casos):
    # Selecciona los casos solicitados por el usuario
    casos = []

    if bloque in {"un-centro", "todos"}:
        casos.extend(("un_centro", caso) for caso in CASOS_UN_CENTRO)

    if bloque in {"dos-centros", "todos"}:
        casos.extend(("dos_centros", caso) for caso in CASOS_DOS_CENTROS)

    if nombres_casos:
        nombres = set(nombres_casos)
        casos = [
            (bloque_exp, caso)
            for bloque_exp, caso in casos
            if caso["nombre"] in nombres
        ]

        faltantes = nombres - {caso["nombre"] for _, caso in casos}
        if faltantes:
            raise ValueError("Casos no encontrados en el bloque seleccionado: " + ", ".join(sorted(faltantes)))

    return casos


def validar_argumentos(args):
    # Comprueba que los argumentos sean válidos
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
    if not 0.0 <= args.mutation_prob <= 1.0:
        raise ValueError("La probabilidad de mutación debe estar en [0,1].")
    if args.k is not None and args.k < 0:
        raise ValueError("k debe ser mayor o igual que 0.")


def main():
    # Ejecuta el experimento completo de Mallows
    args = SimpleNamespace(
        bloque="todos",
        casos=None,
        votantes=VOTANTES_POR_INSTANCIA,
        instancias=INSTANCIAS_POR_CASO,
        ejecuciones=EJECUCIONES_POR_OBJETIVO,
        generations=CONFIGURACION_GENETICO["generations"],
        pop_size=CONFIGURACION_GENETICO["pop_size"],
        mutation_prob=CONFIGURACION_GENETICO["mutation_prob"],
        k=CONFIGURACION_GENETICO["k"],
    )
    validar_argumentos(args)

    configuracion = {
        "nombre": "fija",
        "generations": args.generations,
        "mutation_prob": args.mutation_prob,
        "pop_size": args.pop_size,
        "k": args.k,
    }

    casos = seleccionar_casos(args.bloque, args.casos)
    if not casos:
        raise ValueError("No se ha seleccionado ningún caso experimental.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    resumenes = []
    detalles = []
    posiciones = []

    total_objetivos = sum(len(caso["objetivos"]) for _, caso in casos)
    total_ejecuciones = (
        args.instancias * total_objetivos * args.ejecuciones
    )

    print("Inicio del experimento Mallows", flush=True)
    print(f"Salida: {OUTPUT_DIR}", flush=True)
    print(f"Candidatos: {CANDIDATOS}", flush=True)
    print(f"Votantes por instancia: {args.votantes}", flush=True)
    print(f"Instancias por caso: {args.instancias}", flush=True)
    print(f"Ejecuciones por objetivo: {args.ejecuciones}", flush=True)
    print(f"Configuración AG: {configuracion}", flush=True)
    print(f"Casos seleccionados: {len(casos)}", flush=True)
    print(f"Total de ejecuciones del genético: {total_ejecuciones}", flush=True)

    for caso_idx, (bloque_experimental, caso) in enumerate(casos):
        print()
        print("=" * 88, flush=True)
        print(f"CASO {caso_idx + 1}/{len(casos)}: {caso['nombre']}", flush=True,)
        print(caso["descripcion"], flush=True)
        print(f"Bloques: {caso['bloques']}", flush=True)

        for instancia_idx in range(args.instancias):
            instancia_id = instancia_idx + 1
            instancia_dir = (
                OUTPUT_DIR
                / bloque_experimental
                / caso["nombre"]
                / f"instancia_{instancia_id:02d}"
            )
            seed_instancia = (
                SEED_BASE
                + caso_idx * 10_000
                + instancia_idx
            )

            ranking, metadatos_bloques = generar_ranking_mallows_mixto(candidatos=CANDIDATOS, bloques=caso["bloques"], num_votantes=args.votantes, seed=seed_instancia,)

            matriz = matriz_posiciones(ranking, CANDIDATOS)
            estadisticas = calcular_estadisticas_posicion(matriz=matriz, candidatos=CANDIDATOS, num_votantes=args.votantes,)

            categoria_por_candidato = {
                candidato: categoria
                for categoria, candidato in caso["objetivos"]
            }

            for candidato in CANDIDATOS:
                datos = estadisticas[candidato]
                posiciones.append({"bloque_experimental": bloque_experimental, "caso": caso["nombre"], "descripcion_caso": caso["descripcion"], "instancia": instancia_id, "seed_instancia": seed_instancia, "bloques_mallows": serializar(metadatos_bloques), "candidato": candidato, "posiciones_en_centros": serializar(posiciones_en_centros(candidato, caso["bloques"],)), "posicion_media": datos["posicion_media"], "desviacion_posicion": datos["desviacion_posicion"], "frecuencia_primera": datos["frecuencia_primera"], "frecuencia_ultima": datos["frecuencia_ultima"], "rango_posicion_media": datos["rango_posicion_media"], "categoria_objetivo": categoria_por_candidato.get(candidato, "",),})

            print()
            print(f"Instancia {instancia_id}/{args.instancias} | " f"seed={seed_instancia} | " f"rankings distintos={len(ranking)}", flush=True,)

            for candidato in CANDIDATOS:
                datos = estadisticas[candidato]
                print(f"  {candidato}: media={datos['posicion_media']:.3f}, " f"desv={datos['desviacion_posicion']:.3f}, " f"primero={100 * datos['frecuencia_primera']:.1f}%", flush=True,)

            _pdf_path, svg_path, tex_path = generar_heatmap(matriz=matriz, candidatos=CANDIDATOS, estadisticas=estadisticas, bloque_experimental=bloque_experimental, caso_nombre=caso["nombre"], instancia_id=instancia_id, output_dir=OUTPUT_DIR,)

            print(f"Heatmap SVG: {svg_path}", flush=True)
            print(f"Heatmap TikZ: {tex_path}", flush=True)

            for objetivo_idx, (categoria, candidato) in enumerate(caso["objetivos"]):
                resumen, detalles_objetivo = ejecutar_objetivo(ranking=ranking, bloque_experimental=bloque_experimental, caso=caso, instancia_id=instancia_id, total_instancias_caso=args.instancias, num_votantes=args.votantes, configuracion=configuracion, categoria_objetivo=categoria, candidato_objetivo=candidato, objetivo_idx=objetivo_idx, ejecuciones=args.ejecuciones, seed_instancia=seed_instancia, estadisticas=estadisticas, metadatos_bloques=metadatos_bloques, output_dir=instancia_dir,)

                resumenes.append(resumen)
                detalles.extend(detalles_objetivo)

                # Guarda resultados parciales
                guardar_resultados(resumenes=resumenes, detalles=detalles, posiciones=posiciones, output_dir=OUTPUT_DIR,)

    rutas = guardar_resultados(resumenes=resumenes, detalles=detalles, posiciones=posiciones, output_dir=OUTPUT_DIR,)

    print()
    print("=" * 88, flush=True)
    print("Experimento terminado.", flush=True)
    print(f"Resultados por instancia: {rutas[0]}", flush=True)
    print(f"Ejecuciones detalladas: {rutas[1]}", flush=True)
    print(f"Estadísticas de candidatos: {rutas[2]}", flush=True)
    print(f"Resumen global por ejecuciones: {rutas[3]}", flush=True)
    print(f"Resumen entre instancias: {rutas[4]}", flush=True)
    print(f"Tabla LaTeX global: {rutas[5]}", flush=True)
    print(f"Tabla LaTeX por instancias: {rutas[6]}", flush=True)


if __name__ == "__main__":
    main()
