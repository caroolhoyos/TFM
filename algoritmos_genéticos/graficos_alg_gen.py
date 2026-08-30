"""Generación de gráficos SVG para el historial del algoritmo genético."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Iterable, Mapping


ANCHO = 900
ALTO = 520
MARGEN_IZQUIERDO = 80
MARGEN_DERECHO = 30
MARGEN_SUPERIOR = 25
MARGEN_INFERIOR = 70
NUM_MARCAS = 6


def _numero(valor: object, campo: str) -> float:
    try:
        numero = float(valor)
    except (TypeError, ValueError) as error:
        raise ValueError(f"El campo {campo!r} debe contener un número.") from error

    if not isfinite(numero):
        raise ValueError(f"El campo {campo!r} debe contener un número finito.")
    return numero


def _normalizar_historial(
    historial: Iterable[Mapping[str, object]],
) -> list[dict[str, float]]:
    registros = []
    for indice, registro in enumerate(historial, start=1):
        registros.append(
            {
                "generation": _numero(registro.get("generation", indice), "generation"),
                "best_fitness": _numero(registro.get("best_fitness"), "best_fitness"),
                "average_fitness": _numero(
                    registro.get("average_fitness"), "average_fitness"
                ),
            }
        )

    if not registros:
        raise ValueError("No se puede crear un gráfico con un historial vacío.")
    return registros


def _marcas_x(generaciones: list[float]) -> list[float]:
    if len(generaciones) == 1:
        return generaciones

    indices = {
        round(indice * (len(generaciones) - 1) / (NUM_MARCAS - 1))
        for indice in range(NUM_MARCAS)
    }
    return [generaciones[indice] for indice in sorted(indices)]


def _formatear_marca_x(valor: float) -> str:
    return str(int(valor)) if valor.is_integer() else f"{valor:g}"


def _crear_svg(
    generaciones: list[float],
    valores: list[float],
    destino: Path,
) -> None:
    ancho_grafico = ANCHO - MARGEN_IZQUIERDO - MARGEN_DERECHO
    alto_grafico = ALTO - MARGEN_SUPERIOR - MARGEN_INFERIOR

    x_min = min(generaciones)
    x_max = max(generaciones)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    y_min = min(0.0, min(valores))
    y_max = max(2.0, max(valores))
    if y_min == y_max:
        y_max = y_min + 1.0

    def coordenada_x(valor: float) -> float:
        return MARGEN_IZQUIERDO + (valor - x_min) * ancho_grafico / (x_max - x_min)

    def coordenada_y(valor: float) -> float:
        return MARGEN_SUPERIOR + (y_max - valor) * alto_grafico / (y_max - y_min)

    lineas = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{ANCHO}" '
        f'height="{ALTO}" viewBox="0 0 {ANCHO} {ALTO}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
    ]

    for indice in range(NUM_MARCAS):
        valor = y_min + indice * (y_max - y_min) / (NUM_MARCAS - 1)
        y = coordenada_y(valor)
        lineas.append(
            f'<line x1="{MARGEN_IZQUIERDO}" y1="{y:.2f}" '
            f'x2="{ANCHO - MARGEN_DERECHO}" y2="{y:.2f}" stroke="#e5e7eb" />'
        )
        lineas.append(
            f'<text x="{MARGEN_IZQUIERDO - 12}" y="{y + 4:.2f}" '
            f'text-anchor="end" font-size="12" fill="#4b5563">{valor:.1f}</text>'
        )

    eje_inferior = ALTO - MARGEN_INFERIOR
    lineas.extend(
        [
            f'<line x1="{MARGEN_IZQUIERDO}" y1="{MARGEN_SUPERIOR}" '
            f'x2="{MARGEN_IZQUIERDO}" y2="{eje_inferior}" stroke="#111827" />',
            f'<line x1="{MARGEN_IZQUIERDO}" y1="{eje_inferior}" '
            f'x2="{ANCHO - MARGEN_DERECHO}" y2="{eje_inferior}" stroke="#111827" />',
        ]
    )

    for valor in _marcas_x(generaciones):
        x = coordenada_x(valor)
        lineas.append(
            f'<text x="{x:.2f}" y="{ALTO - 42}" text-anchor="middle" '
            f'font-size="12" fill="#4b5563">{_formatear_marca_x(valor)}</text>'
        )

    puntos = " ".join(
        f"{coordenada_x(generacion):.2f},{coordenada_y(valor):.2f}"
        for generacion, valor in zip(generaciones, valores)
    )
    lineas.extend(
        [
            f'<polyline points="{puntos}" fill="none" stroke="#2563eb" '
            'stroke-width="3" />',
            f'<text x="{ANCHO / 2}" y="{ALTO - 18}" text-anchor="middle" '
            'font-size="14" fill="#111827">Generación</text>',
            f'<text x="18" y="{ALTO / 2}" text-anchor="middle" font-size="14" '
            f'fill="#111827" transform="rotate(-90 18 {ALTO / 2})">Fitness</text>',
            "</svg>",
        ]
    )

    destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def crear_graficos_convergencia(
    historial: Iterable[Mapping[str, object]],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Crea los dos gráficos de convergencia esperados por los experimentos."""
    registros = _normalizar_historial(historial)
    directorio = Path(output_dir)
    directorio.mkdir(parents=True, exist_ok=True)

    generaciones = [registro["generation"] for registro in registros]
    destinos = {
        "mejor_fitness_acumulado": directorio / "mejor_fitness_acumulado.svg",
        "fitness_medio_generacion": directorio / "fitness_medio_generacion.svg",
    }

    _crear_svg(
        generaciones,
        [registro["best_fitness"] for registro in registros],
        destinos["mejor_fitness_acumulado"],
    )
    _crear_svg(
        generaciones,
        [registro["average_fitness"] for registro in registros],
        destinos["fitness_medio_generacion"],
    )
    return destinos
