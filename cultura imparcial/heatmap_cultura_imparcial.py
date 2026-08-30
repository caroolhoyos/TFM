from collections import Counter
from pathlib import Path
from random import Random
import subprocess
import sys 

SRC_DIR = Path(__file__).resolve().parent
MODULOS_DIR = SRC_DIR.parent
if str(MODULOS_DIR) not in sys.path:
    sys.path.insert(0, str(MODULOS_DIR))

def generar_votos_cultura_imparcial(candidatos, num_votantes, seed=123):
    # Genera rankings completos bajo cultura imparcial
    rng = Random(seed)
    votos = []

    for _ in range(num_votantes):
        votos.append(tuple(rng.sample(candidatos, len(candidatos))))

    return votos


def generar_votos_mallows(candidatos, ranking_central, phi, num_votantes, seed=123):
    # Genera rankings con el modelo de Mallows
    if phi < 0 or phi > 1:
        raise ValueError("phi debe estar entre 0 y 1.")

    if set(candidatos) != set(ranking_central) or len(candidatos) != len(ranking_central):
        raise ValueError("El ranking central debe contener exactamente los mismos candidatos.")

    rng = Random(seed)
    votos = []

    for _ in range(num_votantes):
        votos.append(_muestrear_mallows(ranking_central, phi, rng))

    return votos


def _muestrear_mallows(ranking_central, phi, rng):
    # Construye un ranking aleatorio alrededor del ranking central
    ranking = []

    for candidato in ranking_central:
        if phi == 1:
            posicion = rng.randint(0, len(ranking))
        else:
            pesos = []

            for posicion_posible in range(len(ranking) + 1):
                inversiones = len(ranking) - posicion_posible
                pesos.append(phi ** inversiones)

            total = sum(pesos)
            umbral = rng.random() * total
            acumulado = 0
            posicion = len(ranking)

            for i, peso in enumerate(pesos):
                acumulado += peso

                if acumulado >= umbral:
                    posicion = i
                    break

        ranking.insert(posicion, candidato)

    return tuple(ranking)


def matriz_posiciones(votos, candidatos):
    # Cuenta los votos de cada candidato en cada posición
    matriz = {candidato: [0] * len(candidatos) for candidato in candidatos}

    for voto in votos:
        for posicion, candidato in enumerate(voto):
            matriz[candidato][posicion] += 1

    return matriz


def color_azul(valor, minimo, maximo):
    # Calcula un tono azul según el valor observado
    if maximo == minimo:
        t = 0.5
    else:
        t = (valor - minimo) / (maximo - minimo)

    claro = (219, 234, 254)  # #dbeafe
    oscuro = (29, 78, 216)   # #1d4ed8
    t = 0.12 + 0.88 * max(0, min(1, t))
    rgb = []

    for canal_claro, canal_oscuro in zip(claro, oscuro):
        rgb.append(round(canal_claro + (canal_oscuro - canal_claro) * t))

    return tuple(rgb)


def texto_contraste(rgb):
    # Elige texto claro u oscuro según el color de fondo
    r, g, b = [x / 255 for x in rgb]
    luminosidad = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (15, 23, 42) if luminosidad > 0.60 else (248, 250, 252)


def rgb_svg(rgb):
    # Convierte un color RGB al formato usado en SVG
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


def escape_xml(texto):
    # Escapa los caracteres especiales de XML
    return (str(texto) .replace("&", "&amp;") .replace("<", "&lt;") .replace(">", "&gt;") .replace('"', "&quot;"))


def escape_latex(texto):
    # Escapa los caracteres especiales de LaTeX
    reemplazos = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    return "".join(reemplazos.get(caracter, caracter) for caracter in str(texto))


def rgb_latex(rgb):
    # Convierte un color RGB al formato usado en LaTeX
    return f"{rgb[0]},{rgb[1]},{rgb[2]}"


def escribir_tikz(output_path, matriz, candidatos, posiciones_medias=None, cell_w=1.45):
    # Escribe un heatmap individual en formato TikZ
    num_posiciones = len(next(iter(matriz.values())))
    minimo = min(min(fila) for fila in matriz.values())
    maximo = max(max(fila) for fila in matriz.values())
    cell_h = 0.74

    lineas = [
        r"\documentclass[tikz,border=2pt]{standalone}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{xcolor}",
        r"\begin{document}",
        r"\begin{tikzpicture}[x=1cm,y=1cm]",
        r"\definecolor{axisText}{RGB}{15,23,42}",
        r"\definecolor{mutedText}{RGB}{71,85,105}",
        r"\definecolor{lightCellText}{RGB}{248,250,252}",
        r"\definecolor{darkCellText}{RGB}{15,23,42}",
    ]

    for col, candidato in enumerate(candidatos):
        x = col * cell_w + cell_w / 2
        lineas.append(rf"\node[font=\bfseries\small, text=axisText] " rf"at ({x:.3f},0.35) {{{escape_latex(candidato)}}};")
        if posiciones_medias is not None:
            lineas.append(rf"\node[font=\scriptsize, text=mutedText] " rf"at ({x:.3f},0.02) {{{posiciones_medias[candidato]:.3f}}};")

    for posicion in range(num_posiciones):
        y = -posicion * cell_h - 0.35
        lineas.append(rf"\node[font=\bfseries\small, text=axisText, anchor=east] " rf"at (-0.25,{y - cell_h / 2:.3f}) {{{posicion + 1}}};")

        for col, candidato in enumerate(candidatos):
            x = col * cell_w
            valor = matriz[candidato][posicion]
            fondo = color_azul(valor, minimo, maximo)
            texto = texto_contraste(fondo)
            nombre_color = f"celda{posicion}{col}"
            nombre_texto = "darkCellText" if texto == (15, 23, 42) else "lightCellText"

            lineas.append(rf"\definecolor{{{nombre_color}}}{{RGB}}{{{rgb_latex(fondo)}}}")
            lineas.append(rf"\filldraw[fill={nombre_color}, draw=white, line width=0.45pt] " rf"({x:.3f},{y:.3f}) rectangle ({x + cell_w:.3f},{y - cell_h:.3f});")
            lineas.append(rf"\node[font=\bfseries\small, text={nombre_texto}] " rf"at ({x + cell_w / 2:.3f},{y - cell_h / 2:.3f}) {{{valor}}};")

    centro_x = cell_w * len(candidatos) / 2
    centro_y = -cell_h * num_posiciones / 2 - 0.35
    etiqueta_candidato = (
        r"{Candidato (posición media)}"
        if posiciones_medias is not None
        else r"{Candidato}"
    )
    lineas.extend([rf"\node[font=\small, text=mutedText] at ({centro_x:.3f},{-cell_h * num_posiciones - 0.75:.3f}) " rf"{etiqueta_candidato};", rf"\node[font=\small, text=mutedText, rotate=90] at (-0.92,{centro_y:.3f}) " r"{Posición};", r"\end{tikzpicture}", r"\end{document}", "",])

    Path(output_path).write_text("\n".join(lineas), encoding="utf-8")


def escribir_tikz_paneles(output_path, paneles, candidatos):
    # Escribe varios heatmaps en una misma figura TikZ
    cell_w = 1.45
    cell_h = 0.74
    panel_gap = 1.15
    panel_width = cell_w * len(candidatos)
    num_posiciones = len(candidatos)

    lineas = [
        r"\documentclass[tikz,border=2pt]{standalone}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{xcolor}",
        r"\begin{document}",
        r"\begin{tikzpicture}[x=1cm,y=1cm]",
        r"\definecolor{axisText}{RGB}{15,23,42}",
        r"\definecolor{mutedText}{RGB}{71,85,105}",
        r"\definecolor{lightCellText}{RGB}{248,250,252}",
        r"\definecolor{darkCellText}{RGB}{15,23,42}",
    ]

    for panel_idx, panel in enumerate(paneles):
        matriz = panel["matriz"]
        etiqueta = panel.get("etiqueta")
        etiqueta_latex = panel.get("etiqueta_latex")
        minimo = min(min(fila) for fila in matriz.values())
        maximo = max(max(fila) for fila in matriz.values())
        x0 = panel_idx * (panel_width + panel_gap)

        for col, candidato in enumerate(candidatos):
            x = x0 + col * cell_w + cell_w / 2
            lineas.append(rf"\node[font=\bfseries\small, text=axisText] " rf"at ({x:.3f},0.35) {{{escape_latex(candidato)}}};")

        for posicion in range(num_posiciones):
            y = -posicion * cell_h - 0.35
            lineas.append(rf"\node[font=\bfseries\small, text=axisText, anchor=east] " rf"at ({x0 - 0.25:.3f},{y - cell_h / 2:.3f}) {{{posicion + 1}}};")

            for col, candidato in enumerate(candidatos):
                x = x0 + col * cell_w
                valor = matriz[candidato][posicion]
                fondo = color_azul(valor, minimo, maximo)
                texto = texto_contraste(fondo)
                nombre_color = f"panel{panel_idx}celda{posicion}{col}"
                nombre_texto = "darkCellText" if texto == (15, 23, 42) else "lightCellText"

                lineas.append(rf"\definecolor{{{nombre_color}}}{{RGB}}{{{rgb_latex(fondo)}}}")
                lineas.append(rf"\filldraw[fill={nombre_color}, draw=white, line width=0.45pt] " rf"({x:.3f},{y:.3f}) rectangle ({x + cell_w:.3f},{y - cell_h:.3f});")
                lineas.append(rf"\node[font=\bfseries\small, text={nombre_texto}] " rf"at ({x + cell_w / 2:.3f},{y - cell_h / 2:.3f}) {{{valor}}};")

        centro_x = x0 + panel_width / 2
        centro_y = -cell_h * num_posiciones / 2 - 0.35
        lineas.append(rf"\node[font=\small, text=mutedText] at ({centro_x:.3f},{-cell_h * num_posiciones - 0.75:.3f}) " r"{Candidato};")
        lineas.append(rf"\node[font=\small, text=mutedText, rotate=90] at ({x0 - 0.92:.3f},{centro_y:.3f}) " r"{Posición};")

        if etiqueta:
            etiqueta_panel = etiqueta_latex if etiqueta_latex is not None else escape_latex(etiqueta)
            lineas.append(rf"\node[font=\small, text=axisText] at ({centro_x:.3f},{-cell_h * num_posiciones - 1.18:.3f}) " rf"{{{etiqueta_panel}}};")

    lineas.extend([r"\end{tikzpicture}", r"\end{document}", ""])
    Path(output_path).write_text("\n".join(lineas), encoding="utf-8")


def compilar_pdf_latex(tex_path):
    # Compila el archivo TikZ como PDF con LaTeX
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={tex_path.parent}", tex_path.name,], cwd=tex_path.parent, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,)


def crear_heatmap_latex(matriz, candidatos, output_base, posiciones_medias=None, compilar_pdf=True):
    # Genera un heatmap en SVG, TikZ y, opcionalmente, PDF
    output_base = Path(output_base)
    num_posiciones = len(next(iter(matriz.values())))
    minimo = min(min(fila) for fila in matriz.values())
    maximo = max(max(fila) for fila in matriz.values())

    cell_w = max(86, max(len(str(candidato)) for candidato in candidatos) * 7 + 18)
    cell_h = 46
    left = 92
    top = 64 if posiciones_medias is not None else 48
    right = 40
    bottom = 54
    width = left + cell_w * len(candidatos) + right
    height = top + cell_h * num_posiciones + bottom

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    for col, candidato in enumerate(candidatos):
        x = left + col * cell_w
        label_x = x + cell_w / 2
        svg.append(f'<text x="{label_x}" y="{top - 30 if posiciones_medias is not None else top - 18}" ' f'font-family="Latin Modern Roman, Computer Modern, serif" ' f'font-size="13" font-weight="700" text-anchor="middle" fill="#0f172a">' f"{escape_xml(candidato)}</text>")
        if posiciones_medias is not None:
            svg.append(f'<text x="{label_x}" y="{top - 12}" ' f'font-family="Latin Modern Roman, Computer Modern, serif" ' f'font-size="11" text-anchor="middle" fill="#475569">' f"{posiciones_medias[candidato]:.3f}</text>")

    for posicion in range(num_posiciones):
        y = top + posicion * cell_h
        etiqueta = str(posicion + 1)
        svg.append(f'<text x="{left - 18}" y="{y + cell_h / 2 + 5}" ' f'font-family="Latin Modern Roman, Computer Modern, serif" ' f'font-size="12" font-weight="700" text-anchor="end" fill="#334155">{etiqueta}</text>')

        for col, candidato in enumerate(candidatos):
            x = left + col * cell_w
            valor = matriz[candidato][posicion]
            fondo = color_azul(valor, minimo, maximo)
            texto = texto_contraste(fondo)

            svg.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cell_w - 4}" height="{cell_h - 4}" ' f'fill="{rgb_svg(fondo)}" stroke="#ffffff" stroke-width="1.2"/>')
            svg.append(f'<text x="{x + cell_w / 2}" y="{y + cell_h / 2 + 5}" ' f'font-family="Latin Modern Roman, Computer Modern, serif" ' f'font-size="12" font-weight="700" ' f'text-anchor="middle" fill="{rgb_svg(texto)}">{valor}</text>')

    svg.append(f'<text x="28" y="{top + (cell_h * num_posiciones) / 2}" ' f'font-family="Latin Modern Roman, Computer Modern, serif" ' f'font-size="12" font-weight="700" ' f'text-anchor="middle" fill="#475569" ' f'transform="rotate(-90 28 {top + (cell_h * num_posiciones) / 2})">Posición</text>')
    svg.append(f'<text x="{left + (cell_w * len(candidatos)) / 2}" y="{height - 22}" ' f'font-family="Latin Modern Roman, Computer Modern, serif" ' f'font-size="12" font-weight="700" ' f'text-anchor="middle" fill="#475569">' f'{"Candidato (posición media)" if posiciones_medias is not None else "Candidato"}</text>')
    svg.append("</svg>\n")

    svg_path = output_base.with_suffix(".svg")
    tex_path = output_base.with_suffix(".tex")
    pdf_path = output_base.with_suffix(".pdf")
    svg_path.write_text("\n".join(svg), encoding="utf-8")
    escribir_tikz(tex_path, matriz, candidatos, posiciones_medias=posiciones_medias, cell_w=1.45 * cell_w / 86,)
    if compilar_pdf:
        compilar_pdf_latex(tex_path)
    else:
        # Elimina los archivos generados por compilaciones anteriores
        for extension in (".pdf", ".aux", ".log"):
            output_base.with_suffix(extension).unlink(missing_ok=True)

    return pdf_path, svg_path, tex_path


def crear_heatmaps_panel_latex(paneles, candidatos, output_base):
    # Genera una figura con varios heatmaps en fila
    output_base = Path(output_base)
    num_posiciones = len(candidatos)
    cell_w = 86
    cell_h = 46
    left = 92
    top = 48
    right = 40
    bottom = 74
    panel_gap = 50
    panel_width = left + cell_w * len(candidatos) + right
    panel_height = top + cell_h * num_posiciones + bottom
    width = len(paneles) * panel_width + max(0, len(paneles) - 1) * panel_gap
    height = panel_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    for panel_idx, panel in enumerate(paneles):
        matriz = panel["matriz"]
        etiqueta = panel.get("etiqueta")
        minimo = min(min(fila) for fila in matriz.values())
        maximo = max(max(fila) for fila in matriz.values())
        x_offset = panel_idx * (panel_width + panel_gap)

        for col, candidato in enumerate(candidatos):
            x = x_offset + left + col * cell_w
            label_x = x + cell_w / 2
            svg.append(f'<text x="{label_x}" y="{top - 18}" ' f'font-family="Latin Modern Roman, Computer Modern, serif" ' f'font-size="13" font-weight="700" text-anchor="middle" fill="#0f172a">' f"{escape_xml(candidato)}</text>")

        for posicion in range(num_posiciones):
            y = top + posicion * cell_h
            etiqueta_posicion = str(posicion + 1)
            svg.append(f'<text x="{x_offset + left - 18}" y="{y + cell_h / 2 + 5}" ' f'font-family="Latin Modern Roman, Computer Modern, serif" ' f'font-size="12" font-weight="700" text-anchor="end" fill="#334155">' f"{etiqueta_posicion}</text>")

            for col, candidato in enumerate(candidatos):
                x = x_offset + left + col * cell_w
                valor = matriz[candidato][posicion]
                fondo = color_azul(valor, minimo, maximo)
                texto = texto_contraste(fondo)

                svg.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cell_w - 4}" height="{cell_h - 4}" ' f'fill="{rgb_svg(fondo)}" stroke="#ffffff" stroke-width="1.2"/>')
                svg.append(f'<text x="{x + cell_w / 2}" y="{y + cell_h / 2 + 5}" ' f'font-family="Latin Modern Roman, Computer Modern, serif" ' f'font-size="12" font-weight="700" text-anchor="middle" ' f'fill="{rgb_svg(texto)}">{valor}</text>')

        centro_x = x_offset + left + (cell_w * len(candidatos)) / 2
        svg.append(f'<text x="{x_offset + 28}" y="{top + (cell_h * num_posiciones) / 2}" ' f'font-family="Latin Modern Roman, Computer Modern, serif" ' f'font-size="12" font-weight="700" text-anchor="middle" fill="#475569" ' f'transform="rotate(-90 {x_offset + 28} {top + (cell_h * num_posiciones) / 2})">' f"Posición</text>")
        svg.append(f'<text x="{centro_x}" y="{height - 42}" ' f'font-family="Latin Modern Roman, Computer Modern, serif" ' f'font-size="12" font-weight="700" text-anchor="middle" fill="#475569">' f"Candidato</text>")

        if etiqueta:
            svg.append(f'<text x="{centro_x}" y="{height - 14}" ' f'font-family="Latin Modern Roman, Computer Modern, serif" ' f'font-size="12" text-anchor="middle" fill="#0f172a">' f"{escape_xml(etiqueta)}</text>")

    svg.append("</svg>\n")

    svg_path = output_base.with_suffix(".svg")
    tex_path = output_base.with_suffix(".tex")
    pdf_path = output_base.with_suffix(".pdf")
    svg_path.write_text("\n".join(svg), encoding="utf-8")
    escribir_tikz_paneles(tex_path, paneles, candidatos)
    compilar_pdf_latex(tex_path)

    return pdf_path, svg_path, tex_path


def main():
    # Genera ejemplos de cultura imparcial y del modelo de Mallows
    candidatos = ["A", "B", "C", "D", "E"]
    ranking_central = candidatos[:]
    num_votantes = 2000
    seed = 2026
    output_dir = Path(__file__).resolve().parent / "ejemplos"
    output_dir.mkdir(exist_ok=True)

    votos = generar_votos_cultura_imparcial(candidatos, num_votantes, seed)
    matriz = matriz_posiciones(votos, candidatos)

    # Resume los rankings completos generados
    conteo_rankings = Counter(votos)
    print(f"Rankings distintos generados: {len(conteo_rankings)} de {len(candidatos)}! posibles")

    output_base = output_dir / "heatmap_cultura_imparcial_5c_2000v"
    pdf_path, svg_path, tex_path = crear_heatmap_latex(matriz, candidatos, output_base)

    print(f"PDF para LaTeX: {pdf_path}")
    print(f"SVG adicional: {svg_path}")
    print(f"TikZ editable: {tex_path}")

    paneles_mallows_baja_dispersion = []

    for phi, seed_phi in [(0, seed + 10), (0.2, seed + 20)]:
        votos_mallows = generar_votos_mallows(candidatos=candidatos, ranking_central=ranking_central, phi=phi, num_votantes=num_votantes, seed=seed_phi,)
        paneles_mallows_baja_dispersion.append({"etiqueta": f"phi={phi}", "etiqueta_latex": rf"$\phi={phi}$", "matriz": matriz_posiciones(votos_mallows, candidatos),})

    output_mallows_baja = output_dir / "heatmap_mallows_phi_0_02_5c_2000v"
    pdf_path, svg_path, tex_path = crear_heatmaps_panel_latex(paneles_mallows_baja_dispersion, candidatos, output_mallows_baja,)

    print(f"PDF Mallows phi=0 y phi=0.2: {pdf_path}")
    print(f"SVG Mallows phi=0 y phi=0.2: {svg_path}")
    print(f"TikZ Mallows phi=0 y phi=0.2: {tex_path}")

    votos_mallows_phi_1 = generar_votos_mallows(candidatos=candidatos, ranking_central=ranking_central, phi=1, num_votantes=num_votantes, seed=seed + 30,)
    panel_mallows_phi_1 = [{"etiqueta": "phi=1", "etiqueta_latex": r"$\phi=1$", "matriz": matriz_posiciones(votos_mallows_phi_1, candidatos),}]
    output_mallows_phi_1 = output_dir / "heatmap_mallows_phi_1_5c_2000v"
    pdf_path, svg_path, tex_path = crear_heatmaps_panel_latex(panel_mallows_phi_1, candidatos, output_mallows_phi_1,)

    print(f"PDF Mallows phi=1: {pdf_path}")
    print(f"SVG Mallows phi=1: {svg_path}")
    print(f"TikZ Mallows phi=1: {tex_path}")


if __name__ == "__main__":
    main()
