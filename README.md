# TFM 

**Trabajo de Fin de Máster:** *«Dificultad del diseño ad hoc de sistemas electorales para la obtención de un resultado deseado»*

**Autora:** Carolina Hoyos Agudo — <cahoyos@ucm.es>

---


## Estructura del proyecto

```text
TFM/
│
├── algoritmos_genéticos/
│   ├── alg_gen.py                        # Versión base
│   ├── alg_gen_men.py                    # Variante: se detiene al lograr la victoria antes del límite de rondas
│   ├── alg_gen_no_comp.py                # Variante: no compacta el ranking de cada votante
│   └── graficos_alg_gen.py               # Gráficos
│
├── cultura imparcial/                    # Experimento con perfiles sintéticos (modelo de Cultura Imparcial)
│   ├── experimento_IC_1.py
│   ├── experimento_IC_2.py
│   ├── heatmap_cultura_imparcial.py
│   └── resultados/
│
├── mallows/                              # Experimento con perfiles sintéticos (modelo de Mallows)
│   ├── experimento_mallows_1.py
│   └── resultados/
│
├── autonómicas andalucía/               # Experimento con datos reales del CIS (autonómicas Andalucía 2026, P8)
│   ├── experimento_datos_electorales_andalucia_1.py
│   ├── experimento_datos_electorales_andalucia_2.py
│   ├── experimento_datos_electorales_andalucia_3.py
│   └── resultados 1/ · resultados 2/ · resultados 3/
│
├── elecciones generales/                # Experimento con datos reales del CIS (generales 2023, P7)
│   ├── experimento_datos_electorales_generales_1.py
│   ├── experimento_datos_electorales_generales_2.py
│   ├── experimento_datos_electorales_generales_3.py
│   └── resultados 1/ · resultados 3/
│
├── datos_electorales_andalucia_26/      # Datos CIS 3558 
│   └── MD3558/
│
└── datos_electorales_generales_23/      # Datos CIS 3411
    └── 3411csv/
```


---

## Datos 

Los experimentos reales emplean datos del **Centro de Investigaciones
Sociológicas (CIS)**. Estos datos no se pueden redistribuir: deben
solicitarse individualmente al CIS. Por tanto, para poder reproducir los experimentos reales hay
que descargar los siguientes estudios y colocarlos en las rutas indicadas previamente en la estructura:

| Experimento | Estudio CIS | Pregunta | Ruta esperada |
|---|---|---|---|
| Autonómicas Andalucía 2026 | [Preelectoral 3558](https://www.cis.es/es/estudios/preelectoral-elecciones-autonomicas-2026-comunidad-autonoma-de-andalucia) | P8  | `datos_electorales_andalucia_26/MD3558/3558_num.csv` |
| Generales 2023 | [Preelectoral 3411](https://www.cis.es/es/estudios/preelectoral-elecciones-generales-2023) | P7 | `datos_electorales_generales_23/3411csv/3411_num.csv` |


---


### Output

Cada ejecución escribe en su carpeta `resultados*/` los resultados generados en cada ejecución, así como los resultados agregados y algunos gráficos asociados.


---

## Licencia

Distribuido bajo licencia **MIT**. Ver archivo `LICENSE`.
