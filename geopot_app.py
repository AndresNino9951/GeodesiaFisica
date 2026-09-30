"""
Potencial Gravitacional Zonal
=============================

Software de apoyo para el cálculo del potencial gravitacional de un cuerpo
elipsoidal en equilibrio hidrostático, mediante el desarrollo en serie de
armónicos zonales (polinomios de Legendre) según la teoría de Clairaut.

Referencia teórica: Avila, M. "Geodesia Física" (ecuación 2.76 y Tabla 5).

Estructura del programa
------------------------
1. WGS84                 -> Parámetros físicos de referencia.
2. GeodesyEngine         -> Núcleo matemático: toda la física vive aquí,
                            desacoplada de la interfaz.
3. render_*()            -> Funciones de presentación (Streamlit).
4. main()                -> Orquesta la aplicación.

Ejecutar con:
    streamlit run geopot_app.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.special import legendre


# =====================================================================
# 1. CONSTANTES DE REFERENCIA — WGS84, Tabla 3 (Avila, "Geodesia Física")
# =====================================================================

@dataclass(frozen=True)
class WGS84:
    """Parámetros físicos y geométricos del elipsoide de referencia WGS84."""

    Km: float = 3.986004418e14        # Constante kepleriana (m^3/s^2)
    a: float = 6_378_137.0            # Semieje mayor (m)
    b: float = 6_356_752.3142         # Semieje menor (m)
    C: float = 8.0354872e37           # Momento de inercia polar (kg*m^2)
    A: float = 8.0091029e37           # Momento de inercia ecuatorial (kg*m^2)
    M: float = 5.9733328e24           # Masa terrestre (kg)


# =====================================================================
# 2. NÚCLEO MATEMÁTICO — independiente de la interfaz
# =====================================================================

@dataclass
class Armonico:
    """Un término J_{2k} de la serie, con cada paso intermedio de su cálculo."""

    k: int
    grado: int          # 2k
    P: float            # P_{2k}(cos theta)
    signo: int          # (-1)^(k+1)
    coef: float         # 3 e^(2k) / ((2k+1)(2k+3))
    factor: float       # 1 - k + 5k * J0
    J: float            # J_{2k}
    a_r_pot: float       # (a/r)^(2k)
    term: float          # (a/r)^(2k) * J_{2k} * P_{2k}


@dataclass
class ResultadoCalculo:
    """Contenedor con cada paso intermedio del cálculo, para trazabilidad."""

    n: int
    phi_deg: float
    a: float
    b: float
    a2: float
    b2: float
    E2: float
    f: float
    e2: float
    sin2_phi: float
    r: float
    theta_deg: float
    cos_theta: float
    a_sobre_r: float
    J0: float                       # (C - A) / (M E^2)
    armonicos: list[Armonico] = field(default_factory=list)
    term_total: float = 0.0         # suma de todos los términos de la serie
    V_kepler: float = 0.0           # Km / r
    V: float = 0.0


class GeodesyEngine:
    """Encapsula la física del desarrollo del potencial en armónicos zonales.

    Todos los métodos son puros (sin efectos secundarios), lo que permite
    probarlos de forma aislada y reutilizarlos tanto en la interfaz gráfica
    como en un futuro script de línea de comandos o notebook.
    """

    def __init__(self, params: WGS84 = WGS84()):
        self.params = params

    # -- geometría del elipsoide -------------------------------------
    def flattening(self, a: float, b: float) -> float:
        """Achatamiento geométrico f = (a - b) / a."""
        return (a - b) / a

    def eccentricity_sq(self, a: float, b: float) -> float:
        """Excentricidad al cuadrado e^2 = (a^2 - b^2) / a^2."""
        return (a ** 2 - b ** 2) / (a ** 2)

    def radius_at_latitude(self, a: float, f: float, phi_rad):
        """Radio r(phi) = a(1 - f sen^2 phi)."""
        return a * (1.0 - f * np.sin(phi_rad) ** 2)

    # -- coeficientes armónicos ---------------------------------------
    def inertia_ratio(self, E2: float) -> float:
        """H = (C - A) / (M E^2)  — cociente de momentos de inercia.

        Usa E^2 = a^2 - b^2 directamente (no depende de la convención de
        e^2 que se use en la serie), para que coincida siempre con lo
        mostrado en el Paso 7.
        """
        p = self.params
        return (p.C - p.A) / (p.M * E2)

    def zonal_harmonic(self, k: int, e2: float, J0: float):
        """Coeficiente J_2k según la teoría de Clairaut.

        J_2k = (-1)^(k+1) * [3 e^(2k) / ((2k+1)(2k+3))] * [1 - k + 5k * J0]
        """
        signo = (-1) ** (k + 1)
        coef = 3.0 * e2 ** k / ((2 * k + 1) * (2 * k + 3))
        factor = 1 - k + 5 * k * J0
        return signo, coef, factor, signo * coef * factor

    # -- potencial (serie completa 1..n) --------------------------------
    def potential(self, n: int, phi_deg: float, a: float = None, b: float = None) -> ResultadoCalculo:
        """Calcula V(phi) sumando la serie de armónicos zonales J_2 ... J_2n."""
        p = self.params
        a = a if a is not None else p.a
        b = b if b is not None else p.b

        a2, b2 = a ** 2, b ** 2
        E2 = a2 - b2
        f = self.flattening(a, b)
        e2 = self.eccentricity_sq(a, b)

        phi_rad = np.radians(phi_deg)
        sin2_phi = float(np.sin(phi_rad) ** 2)
        r = float(self.radius_at_latitude(a, f, phi_rad))
        theta_deg = 90.0 - phi_deg
        cos_theta = float(np.sin(phi_rad))  # cos(theta) = cos(90-phi) = sin(phi)
        a_sobre_r = a / r

        J0 = self.inertia_ratio(E2)

        armonicos: list[Armonico] = []
        term_total = 0.0
        for k in range(1, n + 1):
            grado = 2 * k
            P = float(legendre(grado)(cos_theta))
            signo, coef, factor, J = self.zonal_harmonic(k, e2, J0)
            a_r_pot = a_sobre_r ** grado
            term = a_r_pot * J * P
            term_total += term
            armonicos.append(Armonico(k, grado, P, signo, coef, factor, J, a_r_pot, term))

        V_kepler = p.Km / r
        V = V_kepler * (1.0 + term_total)

        return ResultadoCalculo(
            n=n, phi_deg=phi_deg, a=a, b=b, a2=a2, b2=b2, E2=E2, f=f, e2=e2,
            sin2_phi=sin2_phi, r=r, theta_deg=theta_deg, cos_theta=cos_theta,
            a_sobre_r=a_sobre_r, J0=J0, armonicos=armonicos,
            term_total=term_total, V_kepler=V_kepler, V=V,
        )

    def potential_at_latitude(self, armonicos: list[Armonico], a: float, b: float, phi_deg: float):
        """V, r, x, z (lon=0) en una latitud dada, sumando la misma serie de armónicos."""
        f = self.flattening(a, b)
        phi_rad = np.radians(phi_deg)
        r = float(self.radius_at_latitude(a, f, phi_rad))
        cos_theta = float(np.sin(phi_rad))
        suma = 0.0
        for arm in armonicos:
            P = float(legendre(arm.grado)(cos_theta))
            suma += (a / r) ** arm.grado * arm.J * P
        V = (self.params.Km / r) * (1.0 + suma)
        x = r * np.cos(phi_rad)
        z = (b / a) * r * np.sin(phi_rad)
        return float(V), r, float(x), float(z)

    def surface_points(self, a: float, b: float, lats_deg, lons_deg):
        """Coordenadas (X, Y, Z) sobre el elipsoide para latitudes/longitudes dadas.

        Acepta escalares o arreglos numpy del mismo tamaño (útil tanto para
        una malla completa como para una sola línea de meridiano/paralelo).
        """
        f = self.flattening(a, b)
        lat_rad = np.radians(lats_deg)
        lon_rad = np.radians(lons_deg)
        r = self.radius_at_latitude(a, f, lat_rad)
        X = r * np.cos(lat_rad) * np.cos(lon_rad)
        Y = r * np.cos(lat_rad) * np.sin(lon_rad)
        Z = (b / a) * r * np.sin(lat_rad)
        return X, Y, Z

    def potential_field(self, armonicos: list[Armonico], a: float, b: float,
                         lats_deg: np.ndarray, lons_deg: np.ndarray):
        """Evalúa V (serie completa) sobre una malla de latitudes/longitudes."""
        f = self.flattening(a, b)
        Lats, Lons = np.meshgrid(lats_deg, lons_deg, indexing="ij")
        lat_rad, lon_rad = np.radians(Lats), np.radians(Lons)

        X, Y, Z = self.surface_points(a, b, Lats, Lons)
        r = self.radius_at_latitude(a, f, lat_rad)

        suma = np.zeros_like(r)
        for arm in armonicos:
            P = legendre(arm.grado)(np.sin(lat_rad))
            suma += (a / r) ** arm.grado * arm.J * P
        V = (self.params.Km / r) * (1.0 + suma)
        return X, Y, Z, V


# =====================================================================
# 3. CAPA DE PRESENTACIÓN (Streamlit)
# =====================================================================

ESTILO = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&display=swap');
:root { --oro: #d4b04c; --oro-suave: rgba(212,176,76,0.28); --texto: #e8edf7; --texto-suave: #a9b7d0; }
.stApp {
    color: var(--texto);
    background:
        linear-gradient(rgba(212,176,76,0.045) 1px, transparent 1px) 0 0 / 56px 56px,
        linear-gradient(90deg, rgba(212,176,76,0.045) 1px, transparent 1px) 0 0 / 56px 56px,
        radial-gradient(ellipse at 15% 0%, #1b2f5c 0%, transparent 55%),
        radial-gradient(ellipse at 100% 100%, #14284d 0%, transparent 50%),
        #0a1122;
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 3rem; }
h1, h2, h3 { font-family: 'Playfair Display', Georgia, 'Times New Roman', serif !important; color: #f4ecd0 !important; }
h1 { font-weight: 700 !important; letter-spacing: 0.3px; border-bottom: 2px solid var(--oro); padding-bottom: 0.4rem; }
h2, h3 { font-weight: 500 !important; border-bottom: 1px solid var(--oro-suave); padding-bottom: 0.25rem; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--texto-suave) !important; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: var(--texto); }
[data-testid="stMarkdownContainer"] strong { color: #f0d98a; }
.katex, .katex * { color: var(--texto); }
.katex-display {
    background: rgba(18,32,64,0.55);
    border: 1px solid rgba(212,176,76,0.16);
    border-left: 3px solid var(--oro);
    border-radius: 6px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0 1rem 0;
    overflow-x: auto;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1c3a 0%, #0a1226 100%);
    border-right: 1px solid var(--oro-suave);
}
[data-testid="stSidebar"] h2 { font-size: 1.25rem; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] label * { color: var(--texto) !important; }
div[data-testid="stColumn"]:nth-of-type(2),
div[data-testid="column"]:nth-of-type(2) { position: sticky; top: 4rem; align-self: flex-start; }
</style>
"""

# -- utilidades de formato LaTeX --------------------------------------
# Regla del curso: ninguna cifra lleva coma de miles; solo el punto decimal.

def num(x: float, dec: int = 2) -> str:
    """Número con punto decimal, sin separador de miles."""
    return f"{x:.{dec}f}"


def sci(x: float, digits: int = 6) -> str:
    """Notación científica en LaTeX: 1.5e-5 -> 1.500000 \\times 10^{-5}."""
    if x == 0:
        return "0"
    mant, exp = f"{x:.{digits}e}".split("e")
    return rf"{mant}\times 10^{{{int(exp)}}}"


def bloque(*lineas: str):
    """Muestra varias líneas alineadas en un solo bloque LaTeX."""
    st.latex(r"\begin{aligned}" + r" \\ ".join(lineas) + r"\end{aligned}")


def paso(k: int, titulo: str):
    st.markdown(f"**Paso {k} — {titulo}**")


# -- piezas de la interfaz -------------------------------------------

def render_sidebar(params: WGS84) -> tuple[int, float]:
    """Dibuja el panel lateral y devuelve (n, phi) ingresados por el usuario."""
    E2 = params.a ** 2 - params.b ** 2
    e2 = E2 / (params.a ** 2)
    with st.sidebar:
        st.markdown("## ⚙️ Parámetros del sistema")
        st.caption("Fijos — WGS84, Tabla 3 (Avila, *Geodesia Física*, pág. 19)")
        st.code(
            f"Km = {params.Km:.6e} m3/s2\n"
            f"a  = {params.a:.4f} m\n"
            f"b  = {params.b:.4f} m\n"
            f"C  = {params.C:.6e} kg.m2\n"
            f"A  = {params.A:.6e} kg.m2\n"
            f"M  = {params.M:.6e} kg\n"
            f"E2 = {E2:.6e} m2\n"
            f"e2 = {e2:.10f}",
            language="text",
        )

        st.markdown("## 🎛️ Entradas")
        n = st.number_input(
            "Número de armónicos n a incluir (se suman J2, J4, ..., J(2n))",
            min_value=1, max_value=8, value=1, step=1,
        )
        phi = st.number_input(
            "Latitud φ (grados)",
            min_value=-90.0, max_value=90.0, value=45.0, format="%g",
        )
        st.divider()
        st.caption("Motor: teoría de Clairaut para el elipsoide de nivel")
        st.caption(
            "Realizado por Andres Felipe Niño Achury — 20221025021  \n"
            "Kevin Alejandro Sánchez Sabogal — 20222025095"
        )
    return int(n), float(phi)


def render_header():
    st.set_page_config(page_title="Potencial Gravitacional Zonal", layout="wide")
    st.markdown(ESTILO, unsafe_allow_html=True)
    st.title("🌍 Potencial Gravitacional Zonal")
    st.caption(
        "Cálculo del potencial gravitacional mediante desarrollo en serie de "
        "armónicos zonales (Legendre), según la teoría de Clairaut para un "
        "elipsoide en equilibrio hidrostático."
    )


def render_procedure(res: ResultadoCalculo, params: WGS84):
    """Desarrollo completo: fórmula -> reemplazo -> resultado, en cada paso."""
    n = res.n
    st.subheader(f"Desarrollo para n = {n}  (se incluyen J2 ... J{2 * n})")

    # 1. Cuadrados de los semiejes
    paso(1, "Cuadrados de los semiejes")
    bloque(
        r"a^2 &= a \cdot a",
        rf"&= ({num(res.a, 4)})^2",
        rf"&= {num(res.a2)}\ \mathrm{{m^2}}",
    )
    bloque(
        r"b^2 &= b \cdot b",
        rf"&= ({num(res.b, 4)})^2",
        rf"&= {num(res.b2)}\ \mathrm{{m^2}}",
    )

    # 2. E^2
    paso(2, "Diferencia de cuadrados")
    bloque(
        r"E^2 &= a^2 - b^2",
        rf"&= {num(res.a2)} - {num(res.b2)}",
        rf"&= {num(res.E2)}\ \mathrm{{m^2}}",
    )

    # 3. Achatamiento
    paso(3, "Achatamiento geométrico")
    bloque(
        r"f &= \dfrac{a-b}{a}",
        rf"&= \dfrac{{{num(res.a, 4)} - {num(res.b, 4)}}}{{{num(res.a, 4)}}}",
        rf"&= {res.f:.10f}",
    )

    # 4. Excentricidad
    paso(4, "Excentricidad al cuadrado")
    bloque(
        r"e^2 &= \dfrac{a^2-b^2}{a^2}",
        rf"&= \dfrac{{{num(res.a2)} - {num(res.b2)}}}{{{num(res.a2)}}}",
        rf"&= {res.e2:.10f}",
    )

    # 5. Radio
    paso(5, "Radio del elipsoide a la latitud φ")
    bloque(
        r"r(\varphi) &= a\,(1 - f\,\mathrm{sen}^2\varphi)",
        rf"&= {num(res.a, 4)}\,\left(1 - {res.f:.8f}\cdot \mathrm{{sen}}^2({res.phi_deg:g}^\circ)\right)",
        rf"&= {num(res.a, 4)}\,\left(1 - {res.f:.8f}\cdot {res.sin2_phi:.8f}\right)",
        rf"&= {num(res.r, 3)}\ \mathrm{{m}}",
    )

    # 6. Colatitud
    paso(6, "Colatitud y coseno de la colatitud")
    bloque(
        r"\theta &= 90^\circ - \varphi",
        rf"&= 90^\circ - ({res.phi_deg:g}^\circ)",
        rf"&= {res.theta_deg:.4f}^\circ",
    )
    bloque(
        r"\cos\theta &= \mathrm{sen}\,\varphi",
        rf"&= \mathrm{{sen}}({res.phi_deg:g}^\circ)",
        rf"&= {res.cos_theta:.8f}",
    )

    # 7. Cociente de momentos de inercia.
    # OJO: esto NO es "J0" — en la serie de armónicos zonales, J0 = 1 por
    # definición (es el término de orden cero, la masa puntual). Este
    # cociente es un valor auxiliar de la fórmula de Clairaut; se llama H
    # para no confundirlo con J0.
    # Se muestra con E^2 = a^2 - b^2 (excentricidad lineal al cuadrado, en
    # m^2) en vez de a^2*e^2 por separado — son la misma cantidad
    # (E^2 = a^2 e^2), solo que esta es la notación que usa el profesor.
    paso(7, "Cociente de momentos de inercia — H")
    p = params
    bloque(
        r"H &= \dfrac{C-A}{M\,E^2}",
        rf"&= \dfrac{{{sci(p.C, 7)} - {sci(p.A, 7)}}}"
        rf"{{({sci(p.M, 7)})\,({sci(res.E2, 7)})}}",
        rf"&= \dfrac{{{sci(p.C - p.A, 7)}}}{{{sci(p.M * res.E2, 7)}}}",
        rf"&= {res.J0:.8f}",
    )

    # 8. Polinomios de Legendre y coeficientes J_2k, uno por armónico
    paso(8, "Polinomios de Legendre y coeficientes zonales J2k")

    st.markdown("Antes de la serie, dos casos que salen directo de la definición, sin necesidad de calcular nada:")
    st.latex(r"J_0 = 1 \qquad \text{(término de orden cero: masa puntual, } P_0(\cos\theta)=1\text{)}")
    st.latex(r"J_1 = 0 \qquad \text{(todo armónico impar es cero, por la simetría norte-sur del elipsoide)}")

    st.markdown(
        r"$P_{2k}(\cos\theta)$ es el polinomio de Legendre de grado $2k$ evaluado en "
        rf"$\cos\theta = {res.cos_theta:.8f}$, y cada coeficiente sigue la fórmula completa de Clairaut:"
    )
    bloque(
        r"J_{2k} &= (-1)^{k+1}\,\dfrac{3\,(e^2)^{k}}{(2k+1)(2k+3)}\,\left[\,1 - k + 5k\,\dfrac{C-A}{M\,E^2}\,\right]"
    )
    st.markdown(
        "Para no repetir esa fracción tan larga en cada uno de los términos de la serie, "
        r"se reemplaza $\dfrac{C-A}{M\,E^2}$ por $H$ (calculado en el Paso 7), quedando:"
    )
    bloque(
        r"J_{2k} &= (-1)^{k+1}\,\dfrac{3\,(e^2)^{k}}{(2k+1)(2k+3)}\,\left[\,1 - k + 5k\,H\,\right]"
    )
    for arm in res.armonicos:
        st.markdown(f"— **k = {arm.k}  (grado {arm.grado}):**")
        bloque(
            rf"P_{{{arm.grado}}}(\cos\theta) &= {arm.P:.8f}",
        )
        bloque(
            rf"J_{{{arm.grado}}} &= (-1)^{{{arm.k}+1}}\,\dfrac{{3\,({res.e2:.10f})^{{{arm.k}}}}}"
            rf"{{({2 * arm.k + 1})({2 * arm.k + 3})}}"
            rf"\,\left[\,1 - {arm.k} + 5\cdot{arm.k}\,H\,\right] \quad (H = {res.J0:.8f})",
            rf"&= ({arm.signo})\,\left({sci(arm.coef)}\right)\,\left({arm.factor:.8f}\right)",
            rf"&= {sci(arm.J)}",
        )

    # 9. Serie del potencial
    paso(9, "Serie del potencial gravitacional V(φ)")
    st.markdown(
        "Por definición, el término de orden cero de la serie es "
        r"$J_0 = 1$ (con $P_0(\cos\theta)=1$): es el término de una masa "
        r"puntual, y da lugar al $Km/r$ que aparece fuera del corchete. "
        "La fórmula general, sumando los armónicos zonales hasta el orden $n$, es:"
    )
    bloque(
        r"V(\varphi) &= \dfrac{Km}{r}\left[\,\underbrace{J_0}_{=\,1} + "
        r"\sum_{k=1}^{n}\left(\dfrac{a}{r}\right)^{2k} J_{2k}\,P_{2k}(\cos\theta)\,\right]"
    )
    st.markdown(f"Con $n={n}$, se reemplaza cada término de la suma:")

    terminos_simb = " + ".join(
        rf"\left(\dfrac{{a}}{{r}}\right)^{{{a.grado}}} J_{{{a.grado}}} P_{{{a.grado}}}(\cos\theta)"
        for a in res.armonicos
    )
    terminos_num = " + ".join(
        rf"({a.a_r_pot:.10f})({sci(a.J)})({a.P:.8f})" for a in res.armonicos
    )
    terminos_val = " + ".join(sci(a.term) for a in res.armonicos)

    bloque(
        rf"V &= \dfrac{{Km}}{{r}}\Big[\,1 + {terminos_simb}\,\Big]",
        rf"&= \dfrac{{{sci(p.Km, 9)}}}{{{num(res.r, 3)}}}\Big[\,1 + {terminos_num}\,\Big]",
        rf"&= \left({sci(res.V_kepler, 9)}\right)\Big[\,1 + {terminos_val}\,\Big]",
        rf"&= \left({sci(res.V_kepler, 9)}\right)\left(1 + {sci(res.term_total)}\right)",
    )
    st.markdown("**Resultado final** (sin notación científica, solo punto decimal):")
    bloque(rf"V &= {num(res.V, 2)}\ \mathrm{{m^2/s^2}}")


def render_reading(res: ResultadoCalculo):
    """Interpretación física del resultado."""
    lat_abs = abs(res.phi_deg)
    zona = "ecuatorial / baja" if lat_abs < 25 else ("latitud media" if lat_abs <= 65 else "polar / alta")
    lista_J = ", ".join(f"J{a.grado}" for a in res.armonicos)
    dV = res.V_kepler * res.term_total
    efecto = "aumenta" if dV > 0 else "disminuye"

    st.subheader("Lectura física del resultado")
    st.markdown(
        f"""
**Serie incluida.** Con $n={res.n}$ se suman los armónicos zonales {lista_J},
cuya corrección conjunta sobre el potencial es de ${sci(res.term_total, 4)}$ (valor relativo).

**Geometría.** A $\\varphi = {res.phi_deg:g}^\\circ$ (zona {zona}) el radio del elipsoide es
$r = {num(res.r, 3)}\\ \\mathrm{{m}}$.

**Potencial.** El término central (masa puntual) aporta
$Km/r = {sci(res.V_kepler, 6)}\\ \\mathrm{{m^2/s^2}}$.
La serie de armónicos {efecto} ese valor en
$\\Delta V = {sci(dV, 4)}\\ \\mathrm{{m^2/s^2}}$,
por lo que el potencial total, sin notación científica, es
$V = {num(res.V, 2)}\\ \\mathrm{{m^2/s^2}}$.
"""
    )


def render_3d(engine: GeodesyEngine, res: ResultadoCalculo, params: WGS84):
    """Elipsoide con aspecto de globo terráqueo: meridianos/paralelos tenues,
    ecuador en rojo y eje polar en azul — como un globo, pero solo con esas
    dos referencias (sin trópicos ni círculos polares)."""
    st.subheader(f"Visualización 3D — serie hasta n = {res.n}")
    a, b = params.a, params.b
    lats = np.linspace(-90, 90, 45)
    lons = np.linspace(-180, 180, 45)
    X, Y, Z, V = engine.potential_field(res.armonicos, a, b, lats, lons)
    V_mj = V / 1e6

    v_eval, _, x_eval, z_eval = engine.potential_at_latitude(res.armonicos, a, b, res.phi_deg)
    v_eq, _, _, _ = engine.potential_at_latitude(res.armonicos, a, b, 0.0)
    v_pn, _, _, z_pn = engine.potential_at_latitude(res.armonicos, a, b, 90.0)
    v_ps, _, _, z_ps = engine.potential_at_latitude(res.armonicos, a, b, -90.0)

    fig = go.Figure(go.Surface(
        x=X / 1e3, y=Y / 1e3, z=Z / 1e3,
        surfacecolor=V_mj, colorscale="Cividis", opacity=0.55,
        showscale=False,
        customdata=V_mj,
        hovertemplate="Potencial: %{customdata:.4f} MJ/kg<extra></extra>",
        name="Elipsoide",
    ))

    # Malla de meridianos y paralelos (solo de referencia visual, como un globo)
    color_malla = "rgba(224,230,245,0.45)"
    lat_linea = np.linspace(-90, 90, 61)
    for lon0 in range(-180, 180, 30):
        Xm, Ym, Zm = engine.surface_points(a, b, lat_linea, np.full_like(lat_linea, lon0))
        fig.add_trace(go.Scatter3d(
            x=Xm / 1e3, y=Ym / 1e3, z=Zm / 1e3, mode="lines",
            line=dict(color=color_malla, width=1.5),
            hoverinfo="skip", showlegend=False,
        ))
    lon_linea = np.linspace(-180, 180, 91)
    for lat0 in (-60, -30, 30, 60):
        Xp, Yp, Zp = engine.surface_points(a, b, np.full_like(lon_linea, lat0), lon_linea)
        fig.add_trace(go.Scatter3d(
            x=Xp / 1e3, y=Yp / 1e3, z=Zp / 1e3, mode="lines",
            line=dict(color=color_malla, width=1.5),
            hoverinfo="skip", showlegend=False,
        ))

    # Ecuador, destacado en rojo (única "línea" de referencia con nombre)
    Xeq, Yeq, Zeq = engine.surface_points(a, b, np.zeros_like(lon_linea), lon_linea)
    fig.add_trace(go.Scatter3d(
        x=Xeq / 1e3, y=Yeq / 1e3, z=Zeq / 1e3, mode="lines",
        line=dict(color="#e5342a", width=6), name="Ecuador",
        customdata=np.full(len(lon_linea), v_eq / 1e6),
        hovertemplate="<b>Ecuador</b><br>Potencial: %{customdata:.4f} MJ/kg<extra></extra>",
    ))

    # Ejes X, Y, Z sobresaliendo un poco de la superficie, como varillas
    # que atraviesan el elipsoide de lado a lado (igual que el eje polar).
    lim = (a / 1e3) * 1.12
    z_top, z_bot = (z_pn / 1e3) * 1.12, (z_ps / 1e3) * 1.12
    fig.add_trace(go.Scatter3d(
        x=[-lim, lim], y=[0, 0], z=[0, 0], mode="lines",
        line=dict(color="#3fbf7f", width=8), name="Eje X", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[-lim, lim], z=[0, 0], mode="lines",
        line=dict(color="#a86ee0", width=8), name="Eje Y", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[z_bot, z_top], mode="lines",
        line=dict(color="#3f6fd1", width=9), name="Eje polar (Z)", hoverinfo="skip",
    ))
    for nombre, z, v_mj in (("Polo Norte", z_pn, v_pn / 1e6), ("Polo Sur", z_ps, v_ps / 1e6)):
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[z / 1e3], mode="markers",
            marker=dict(size=6, color="#3f6fd1"), name=nombre,
            customdata=[v_mj],
            hovertemplate=f"<b>{nombre}</b><br>Potencial: %{{customdata:.4f}} MJ/kg<extra></extra>",
        ))

    # Punto evaluado (la latitud elegida en el panel lateral)
    fig.add_trace(go.Scatter3d(
        x=[x_eval / 1e3], y=[0], z=[z_eval / 1e3], mode="markers",
        marker=dict(size=9, color="#f2a71b"), name="Punto evaluado",
        customdata=[v_eval / 1e6],
        hovertemplate="<b>Punto evaluado</b><br>Potencial: %{customdata:.4f} MJ/kg<extra></extra>",
    ))

    ejes = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(212,176,76,0.18)",
                zerolinecolor="rgba(212,176,76,0.35)", color="#c9d4ea")
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X (km)", **ejes),
            yaxis=dict(title="Y (km)", **ejes),
            zaxis=dict(title="Z (km)", **ejes),
            aspectmode="data",
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8edf7"),
        margin=dict(l=0, r=0, b=0, t=10), height=560,
        legend=dict(orientation="h", y=-0.05),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "El elipsoide se dibuja como un globo, con su malla de meridianos y paralelos. "
        "El ecuador (rojo) y los ejes X, Y, Z (verde, morado, azul) sobresalen de la "
        "superficie como referencia. Pasa el cursor sobre el elipsoide, el ecuador, "
        "los polos o el punto evaluado para ver el potencial gravitacional en MJ/kg."
    )


# =====================================================================
# 4. ORQUESTACIÓN
# =====================================================================

def main():
    render_header()
    params = WGS84()
    engine = GeodesyEngine(params)

    n, phi = render_sidebar(params)
    resultado = engine.potential(n, phi)

    col_calculo, col_3d = st.columns([1.1, 1], gap="large")
    with col_calculo:
        render_procedure(resultado, params)
        render_reading(resultado)
    with col_3d:
        render_3d(engine, resultado, params)


if __name__ == "__main__":
    main()
