"""
calentamiento.py
================
Modelo de Kuramoto sobre grafos K_n (completo) y C_n (ciclo).
Tres modos de ejecución:
  [B] Bosquejo  – muestra matrices, espectro, un paso de Euler.
  [S] Simulación – corre el bucle completo, grafica |r|(t) y E(t).
  [A] Animación  – simula y muestra los nodos moviéndose en el círculo.

Ejercicio 2: ciclo C_n, twisted states, Hessiano, estabilidad.

Tecnología: Python 3, numpy, matplotlib.  Un solo archivo, sin clases.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ──────────────────────────────────────────────────────────────
# Funciones del modelo: grafos
# ──────────────────────────────────────────────────────────────

def crear_grafo_K(n):
    """Crea la matriz de adyacencia del grafo completo K_n.
    A[i,j] = 1 si i != j, 0 en la diagonal."""
    A = np.ones((n, n), dtype=float) - np.eye(n)
    return A


def crear_grafo_C(n):
    """Crea la matriz de adyacencia del ciclo C_n.
    Ciclo: grado 2, anillo cerrado.
    Cada nodo v conectado con (v+1) mod n y (v-1) mod n."""
    A = np.zeros((n, n), dtype=float)
    for v in range(n):
        A[v, (v + 1) % n] = 1.0
        A[v, (v - 1) % n] = 1.0
    return A


def crear_grafo_Cnk(n, k):
    """Crea la matriz de adyacencia del grafo C_{n,k}.
    C_{n,k}: grado 2k, k vecinos por lado. C_{n,1} = C_n.
    Para n impar y k=(n-1)/2, C_{n,k} = K_n.
    Cada nodo v conectado con v ± 1, v ± 2, ..., v ± k (mod n)."""
    k_max = (n - 1) // 2
    if k < 1 or k > k_max:
        raise ValueError(f"k debe satisfacer 1 <= k <= {k_max} (para n={n}).")
    A = np.zeros((n, n), dtype=float)
    for v in range(n):
        for j in range(1, k + 1):
            A[v, (v + j) % n] = 1.0
            A[v, (v - j) % n] = 1.0
    return A


# ──────────────────────────────────────────────────────────────
# Funciones del modelo: dinámica y observables
# ──────────────────────────────────────────────────────────────

def laplaciano(A):
    """Calcula la matriz de grados D y el laplaciano L = D - A.
    D es diagonal con D[i,i] = grado del nodo i."""
    D = np.diag(A.sum(axis=1))
    L = D - A
    return D, L


def velocidad(theta, A):
    """Calcula d(theta)/dt para cada oscilador.
    dtheta[v] = sum_u A[v,u] * sin(theta[u] - theta[v]).
    Loop explícito sobre v (sin vectorización)."""
    n = len(theta)
    dtheta = np.zeros(n)
    for v in range(n):
        suma = 0.0
        for u in range(n):
            suma += A[v, u] * np.sin(theta[u] - theta[v])
        dtheta[v] = suma
    return dtheta


def parametro_orden(theta):
    """Calcula el parámetro de orden complejo r = mean(exp(i*theta)).
    Devuelve el número complejo r."""
    r = np.mean(np.exp(1j * theta))
    return r


def energia_K(theta, n):
    """Calcula la energía del modelo de Kuramoto en K_n.
    E = (n^2 / 2) * (1 - |r|^2).
    Solo válida en K_n (grafo completo)."""
    r = parametro_orden(theta)
    E = (n**2 / 2) * (1 - abs(r)**2)
    return E


def energia_grafo(theta, A):
    """Calcula la energía general del modelo de Kuramoto en un grafo arbitrario.
    E = Σ_{aristas} (1 - cos(θ_u - θ_v)). Válida para cualquier grafo (K_n, C_n, etc.)."""
    n = len(theta)
    E = 0.0
    for v in range(n):
        for u in range(v + 1, n):
            if A[v, u] > 0:
                # suma sobre aristas de (1 - cos), igual que en papel
                E += A[v, u] * (1.0 - np.cos(theta[u] - theta[v]))
    return E


def espectro_laplaciano(L):
    """Calcula autovalores y autovectores del laplaciano L.
    Usa np.linalg.eigh (matriz simétrica real).
    eigh ordena ascendente; Fiedler = autovalores[1]."""
    autovalores, autovectores = np.linalg.eigh(L)
    return autovalores, autovectores


def paso_euler(theta, A, dt):
    """Realiza un paso de Euler explícito.
    theta_nuevo = theta + dt * velocidad(theta, A)."""
    dtheta = velocidad(theta, A)
    theta_nuevo = theta + dt * dtheta
    return theta_nuevo, dtheta


# ──────────────────────────────────────────────────────────────
# Verificación de punto crítico
# ──────────────────────────────────────────────────────────────

def verificar_punto_critico(theta, A):
    """Calcula dtheta y su norma para verificar si theta es punto crítico.
    Para el twisted state exacto, la norma debe ser ~0."""
    dtheta = velocidad(theta, A)
    norma = np.linalg.norm(dtheta)
    es_critico = norma < 1e-10
    print(f"\n{'='*50}")
    print("  Verificación de punto crítico")
    print(f"{'='*50}")
    print(f"  ||dtheta|| = {norma:.2e} (¿es punto crítico? {norma < 1e-10})")
    if es_critico:
        print("  → ||dtheta|| ≈ 0: punto crítico verificado ✔")
    return dtheta, norma


# ──────────────────────────────────────────────────────────────
# Hessiano y análisis de estabilidad
# ──────────────────────────────────────────────────────────────

def hessiano(theta, A):
    """Calcula el Hessiano del funcional de energía en theta.
    Laplaciano con pesos cos(theta_j - theta_k).
    H[j,k] = -A[j,k] * cos(theta[j] - theta[k]) para j != k.
    H[j,j] = sum_k A[j,k] * cos(theta[j] - theta[k])."""
    n = len(theta)
    H = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            cjk = A[j, k] * np.cos(theta[j] - theta[k])
            if j != k:
                H[j, k] = -cjk
            else:
                # diagonal: se suma en el loop de abajo
                pass
        # diagonal = suma de los pesos de las aristas
        H[j, j] = sum(A[j, k] * np.cos(theta[j] - theta[k]) for k in range(n))
    return H


def analizar_estabilidad(H, umbral=1e-8):
    """Analiza la estabilidad del punto crítico a partir del Hessiano H.
    Estabilidad = Hessiano positivo sobre 1-perp (excluyendo la rotación global).
    es_estable = todos los autovalores[1:] > umbral."""
    autovalores, autovectores = np.linalg.eigh(H)
    fiedler_H = autovalores[1]
    # El primer autovalor (~0) corresponde a la rotación global
    es_estable = np.all(autovalores[1:] > umbral)

    print(f"\n{'='*50}")
    print("  Análisis de estabilidad (Hessiano)")
    print(f"{'='*50}")
    for i, lam in enumerate(autovalores):
        nota = ""
        if i == 0:
            nota = "  (rotación global, no cuenta)"
        elif i == 1:
            signo = "positivo → estable" if lam > umbral else (
                "cero → marginal" if abs(lam) <= umbral else "negativo → INESTABLE")
            nota = f"  ◀ Fiedler ({signo})"
        print(f"  μ_{i} = {lam:.6f}{nota}")

    if es_estable:
        print(f"\n  ✔ Mínimo local ESTABLE (todos μ_i > 0 para i ≥ 1)")
    else:
        negativos = np.sum(autovalores[1:] <= umbral)
        print(f"\n  ✘ NO es mínimo estable ({negativos} autovalor(es) ≤ 0 para i ≥ 1)")

    return autovalores, autovectores, es_estable, fiedler_H


def verificar_relacion_twisted(H, L, n, q):
    """Verifica que H = cos(2πq/n) · L para el twisted state.
    En el twisted state todas las diferencias valen 2πq/n,
    el cos es común, H = c·L."""
    c = np.cos(2 * np.pi * q / n)
    diferencia = np.linalg.norm(H - c * L)

    print(f"\n{'='*50}")
    print("  Verificación H = cos(2πq/n) · L")
    print(f"{'='*50}")
    print(f"  c = cos(2π·{q}/{n}) = {c:.6f}")
    print(f"  ||H - c·L|| = {diferencia:.2e}  "
          f"({'✔ se cumple' if diferencia < 1e-10 else '✘ NO se cumple'})")

    # Comparar autovalores
    autovalores_H = np.linalg.eigh(H)[0]
    autovalores_L = np.linalg.eigh(L)[0]
    print(f"\n  Autovalores de H = c * autovalores de L:")
    print(f"  {'i':>3}  {'μ_i (H)':>12}  {'c·λ_i (L)':>12}  {'diff':>10}")
    print(f"  {'─'*3}  {'─'*12}  {'─'*12}  {'─'*10}")
    for i in range(n):
        h_i = autovalores_H[i]
        cl_i = c * autovalores_L[i]
        print(f"  {i:3d}  {h_i:12.6f}  {cl_i:12.6f}  {abs(h_i - cl_i):10.2e}")

    return c, diferencia


def encontrar_twisted_estables(n):
    """Encuentra numéricamente los estados twisted estables θ_v = 2πqv/n en C_n.
    Para cada q de 1 a n-1, calcula el Hessiano, sus autovalores y determina
    si es un mínimo local estable (todos los μ_i > 0 para i ≥ 1).
    Devuelve la lista de q estables."""
    A = crear_grafo_C(n)
    print(f"\n{'='*55}")
    print(f"  Búsqueda numérica de estados twisted estables en C_{n}")
    print(f"{'='*55}")
    print(f"  {'q':>3}  {'cos(2πq/n)':>12}  {'μ_1 (Fiedler)':>14}  {'Estabilidad':>12}")
    print(f"  {'─'*3}  {'─'*12}  {'─'*14}  {'─'*12}")

    q_estables = []
    for q_cand in range(1, n):
        theta_cand = np.array([2 * np.pi * q_cand * v / n for v in range(n)])
        H_cand = hessiano(theta_cand, A)
        autov_H = np.linalg.eigh(H_cand)[0]
        c = np.cos(2 * np.pi * q_cand / n)
        fiedler_cand = autov_H[1]
        es_est = np.all(autov_H[1:] > 1e-8)
        if es_est:
            q_estables.append(q_cand)
            estado_str = "✔ ESTABLE"
        else:
            estado_str = "✘ Inestable"
        print(f"  {q_cand:3d}  {c:12.4f}  {fiedler_cand:14.4f}  {estado_str:>12}")

    print(f"\n  → Estados twisted estables encontrados: q ∈ {q_estables}")
    print(f"    (Criterio teórico: cos(2πq/n) > 0  ⟺  |q| < n/4)")
    return q_estables


# ──────────────────────────────────────────────────────────────
# Gráficos del espectro del Hessiano
# ──────────────────────────────────────────────────────────────

def graficar_espectro(autovalores, titulo):
    """Gráfico de barras de los autovalores del Hessiano ordenados.
    Positivos en azul, negativos en rojo, ~0 en gris.
    Anota el Fiedler (λ_1) con su valor."""
    n = len(autovalores)
    indices = np.arange(n)

    # Imprimir en consola junto al gráfico
    print(f"\n{'='*50}")
    print(f"  Espectro del Hessiano — {titulo}")
    print(f"{'='*50}")
    print(f"  λ_0 = {autovalores[0]:.6f} (rotación global, no cuenta)")
    if n >= 2:
        fiedler = autovalores[1]
        estado_f = "positivo ⇒ estable" if fiedler > 1e-8 else (
            "cero ⇒ marginal" if abs(fiedler) <= 1e-8 else "negativo ⇒ inestable")
        print(f"  λ_1 = {fiedler:.6f} (Fiedler: {estado_f})")

    # Colores por signo
    colores = []
    for lam in autovalores:
        if abs(lam) < 1e-8:
            colores.append('#999999')  # gris para ~0
        elif lam > 0:
            colores.append('#2196F3')  # azul para positivos
        else:
            colores.append('#F44336')  # rojo para negativos

    fig, ax = plt.subplots(figsize=(8, 5))
    barras = ax.bar(indices, autovalores, color=colores, edgecolor='white',
                    linewidth=0.5, zorder=3)
    ax.axhline(0, color='black', linewidth=0.8, zorder=2)
    ax.set_xlabel("Índice i", fontsize=11)
    ax.set_ylabel("Autovalor", fontsize=11)
    ax.set_title(f"Espectro del Hessiano — {titulo}",
                 fontsize=13, fontweight='bold')
    ax.set_xticks(indices)
    ax.set_xticklabels([f"λ_{i}" for i in indices], fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Anotar el Fiedler (λ_1)
    if n >= 2:
        fiedler = autovalores[1]
        offset = max(abs(autovalores)) * 0.15 if max(abs(autovalores)) > 0 else 0.5
        ax.annotate(f"Fiedler = {fiedler:.4f}",
                    xy=(1, fiedler),
                    xytext=(1.5, fiedler + offset),
                    fontsize=10, fontweight='bold', color='#0D47A1',
                    arrowprops=dict(arrowstyle='->', color='#0D47A1', lw=1.5),
                    zorder=5)

    # Leyenda manual
    from matplotlib.patches import Patch
    leyenda = [Patch(color='#2196F3', label='Positivo (estable)'),
               Patch(color='#F44336', label='Negativo (inestable)'),
               Patch(color='#999999', label='≈ 0 (rotación global)')]
    ax.legend(handles=leyenda, loc='upper right', fontsize=9)

    plt.tight_layout()
    return fig


def graficar_comparacion_espectros(autovalores_K, autovalores_C, n_K, n_C, q=1):
    """Compara en dos subplots lado a lado el espectro del Hessiano de K_n vs C_n.
    Muestra que en K_n todos los autovalores internos son grandes (pozo profundo),
    mientras que en C_n están escalados por c = cos(2πq/n) (pozo más débil)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    def dibujar_barras(ax, autovalores, titulo, subtitulo=""):
        n = len(autovalores)
        indices = np.arange(n)
        colores = ['#999999' if abs(lam) < 1e-8 else ('#2196F3' if lam > 0 else '#F44336')
                   for lam in autovalores]
        ax.bar(indices, autovalores, color=colores, edgecolor='white', linewidth=0.5, zorder=3)
        ax.axhline(0, color='black', linewidth=0.8, zorder=2)
        ax.set_xlabel("Índice i", fontsize=10)
        ax.set_ylabel("Autovalor μ_i", fontsize=10)
        ax.set_title(titulo, fontsize=11, fontweight='bold')
        ax.set_xticks(indices)
        ax.set_xticklabels([f"λ_{i}" for i in indices], fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        if n >= 2:
            fiedler = autovalores[1]
            offset_y = max(abs(autovalores)) * 0.12 if max(abs(autovalores)) > 0 else 0.5
            ax.annotate(f"Fiedler = {fiedler:.3f}",
                        xy=(1, fiedler),
                        xytext=(1.3, fiedler + offset_y),
                        fontsize=9, fontweight='bold', color='#0D47A1',
                        arrowprops=dict(arrowstyle='->', color='#0D47A1', lw=1.2),
                        zorder=5)
        if subtitulo:
            ax.text(0.04, 0.90, subtitulo, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', alpha=0.85))

    c = np.cos(2 * np.pi * q / n_C)
    dibujar_barras(ax1, autovalores_K, f"K_{n_K} (sincronía global)",
                   "Autovalores internos altos\nPozo profundo (sincroniza siempre)")
    dibujar_barras(ax2, autovalores_C, f"C_{n_C} (twisted q={q})",
                   f"Escalados por c = cos(2π·{q}/{n_C}) = {c:.3f}\nPozo más débil (mínimo local espurio)")

    from matplotlib.patches import Patch
    leyenda = [Patch(color='#2196F3', label='Positivo (estable)'),
               Patch(color='#F44336', label='Negativo (inestable)'),
               Patch(color='#999999', label='≈ 0 (rotación global)')]
    fig.legend(handles=leyenda, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=3, fontsize=9)

    plt.suptitle("K_n vs C_n: espectro del Hessiano", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig



# ──────────────────────────────────────────────────────────────
# Funciones auxiliares de impresión
# ──────────────────────────────────────────────────────────────

def rad_a_grados(theta):
    """Convierte un arreglo de radianes a grados sexagesimales."""
    return np.degrees(theta)


def imprimir_matriz(nombre, M):
    """Imprime una matriz con su nombre."""
    print(f"\n{'='*50}")
    print(f"  {nombre}")
    print(f"{'='*50}")
    print(M)


def imprimir_autovalores(autovalores):
    """Imprime los autovalores con índice y resalta el de Fiedler."""
    print(f"\n{'='*50}")
    print("  Autovalores del Laplaciano")
    print(f"{'='*50}")
    for i, lam in enumerate(autovalores):
        marca = "  ◀ Fiedler" if i == 1 else ""
        print(f"  lambda_{i} = {lam:.6f}{marca}")

    # Advertencia si el grafo está desconectado (más de un autovalor ~0)
    ceros = np.sum(np.abs(autovalores) < 1e-10)
    if ceros > 1:
        print(f"\n  ⚠  ADVERTENCIA: lambda_1 ≈ 0 → grafo desconectado "
              f"(hay {ceros} autovalores nulos).")


def graficar_circulo(theta, titulo):
    """Dibuja los osciladores sobre el círculo unitario.
    Cada nodo se ubica en el ángulo theta[i]."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Círculo unitario
    angulos_circulo = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(angulos_circulo), np.sin(angulos_circulo),
            color='#cccccc', linewidth=1, zorder=1)

    # Nodos
    x = np.cos(theta)
    y = np.sin(theta)
    ax.scatter(x, y, s=100, c='#2196F3', edgecolors='#0D47A1',
               linewidths=1.5, zorder=3)

    # Etiquetas: índice y ángulo en grados
    grados = rad_a_grados(theta)
    for i in range(len(theta)):
        # Desplazamiento radial para la etiqueta
        offset = 0.15
        ax.annotate(f"{i}\n({grados[i]:.1f}°)",
                     xy=(x[i], y[i]),
                     xytext=((1 + offset) * x[i], (1 + offset) * y[i]),
                     fontsize=8, ha='center', va='center',
                     arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                     zorder=4)

    # Líneas radiales desde el origen a cada nodo
    for i in range(len(theta)):
        ax.plot([0, x[i]], [0, y[i]], color='#90CAF9', linewidth=0.8, zorder=2)

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    ax.set_title(titulo, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# Pedir dt y T al usuario (compartido por Simulación y Animación)
# ──────────────────────────────────────────────────────────────

def pedir_dt_y_T(autovalores):
    """Pide dt y T al usuario con advertencia de estabilidad.
    Devuelve (dt, T, pasos)."""
    lambda_max = autovalores[-1]
    dt_critico = 2.0 / lambda_max if lambda_max > 0 else 1.0

    print(f"\n  lambda_max = {lambda_max:.4f}")
    print(f"  dt crítico (2/lambda_max) = {dt_critico:.4f}")

    # Pedir dt
    while True:
        entrada = input(f"dt (Enter = 0.01): ").strip()
        if entrada == '':
            dt = 0.01
        else:
            try:
                dt = float(entrada)
                if dt <= 0:
                    print("  dt debe ser > 0.")
                    continue
            except ValueError:
                print("  Ingresá un número.")
                continue
        # Advertencia si dt >= dt_critico
        if dt >= dt_critico:
            print(f"  ⚠  dt={dt} >= 2/lambda_max={dt_critico:.4f}. Euler puede ser inestable.")
            conf = input("  ¿Continuar igual? (s/N): ").strip().lower()
            if conf != 's':
                continue
        break

    # Pedir T (tiempo total)
    while True:
        entrada = input(f"T (tiempo total, Enter = 10): ").strip()
        if entrada == '':
            T = 10.0
        else:
            try:
                T = float(entrada)
                if T <= 0:
                    print("  T debe ser > 0.")
                    continue
            except ValueError:
                print("  Ingresá un número.")
                continue
        break

    pasos = int(T / dt)
    print(f"  → dt={dt}, T={T}, pasos={pasos}")
    return dt, T, pasos


# ──────────────────────────────────────────────────────────────
# Función de energía adaptativa (usa E_K para K_n, E_grafo para C_n)
# ──────────────────────────────────────────────────────────────

def calcular_energia(theta, A, n, tipo_grafo):
    """Elige la fórmula de energía según el tipo de grafo:
    E = (n²/2)(1 - |r|²) para K_n, o E = Σ_{aristas} (1 - cos(θ_u - θ_v)) para grafos generales."""
    if tipo_grafo == 'K':
        return energia_K(theta, n)
    else:
        return energia_grafo(theta, A)


# ──────────────────────────────────────────────────────────────
# Modo Simulación: bucle completo, grafica |r|(t) y E(t)
# ──────────────────────────────────────────────────────────────

def simular(theta, A, n, dt, pasos, tipo_grafo='K', k_vecinos=None):
    """Corre la simulación completa y grafica |r|(t) y E(t).
    Corte temprano si |r| > 0.999 (ya sincronizó) — solo para K_n.
    Para C_n y C_{n,k} no hay corte (puede no sincronizar).
    Devuelve (theta_final, tiempos, r_abs_hist, E_hist)."""
    theta_actual = theta.copy()

    # Historiales para graficar (listas, se recortan al final)
    tiempos = [0.0]
    r_abs_hist = [abs(parametro_orden(theta_actual))]
    E_hist = [calcular_energia(theta_actual, A, n, tipo_grafo)]

    # Bucle de simulación con corte temprano (solo K_n)
    print("\n  Simulando...")
    for k in range(1, pasos + 1):
        theta_actual, _ = paso_euler(theta_actual, A, dt)
        r_k = parametro_orden(theta_actual)
        abs_r_k = abs(r_k)
        r_abs_hist.append(abs_r_k)
        E_hist.append(calcular_energia(theta_actual, A, n, tipo_grafo))
        tiempos.append(k * dt)

        # Corte temprano solo para K_n (siempre sincroniza)
        if tipo_grafo == 'K' and abs_r_k > 0.999:
            print(f"  Corte temprano en paso {k} (t={k*dt:.2f}): "
                  f"|r| = {abs_r_k:.6f} > 0.999")
            break

    # Convertir a arrays
    tiempos = np.array(tiempos)
    r_abs_hist = np.array(r_abs_hist)
    E_hist = np.array(E_hist)

    print(f"  Simulación completa ({len(tiempos)} pasos, t_final={tiempos[-1]:.2f}).")

    # Etiqueta del grafo
    if tipo_grafo == 'K':
        etiq = f"K$_{{{n}}}$"
    elif tipo_grafo == 'D' and k_vecinos is not None:
        etiq = f"C$_{{{n},{k_vecinos}}}$"
    else:
        etiq = f"C$_{{{n}}}$"

    # Graficar |r|(t) y E(t)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(tiempos, r_abs_hist, color='#2196F3', linewidth=1.5)
    ax1.set_xlabel("t")
    ax1.set_ylabel("|r|")
    ax1.set_title("|r|(t) – parámetro de orden", fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(1.0, color='gray', linewidth=0.5, linestyle='--')

    ax2.plot(tiempos, E_hist, color='#F44336', linewidth=1.5)
    ax2.set_xlabel("t")
    ax2.set_ylabel("E")
    ax2.set_title("E(t) – energía", fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Simulación Kuramoto {etiq}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    return theta_actual, tiempos, r_abs_hist, E_hist


# ──────────────────────────────────────────────────────────────
# Modo Animación: FuncAnimation con gridspec (círculo + curvas)
# ──────────────────────────────────────────────────────────────

def animar(theta, A, n, dt, pasos, semilla, tipo_grafo='K',
           cada_cuantos=1, interval=100, cond_inicial='R', q=None, k_vecinos=None):
    """Precalcula la simulación completa y la muestra como animación.
    Layout (gridspec 2 filas):
      Fila 0 (arriba, 2 columnas): círculo unitario con nodos + flecha r.
      Fila 1, col 0: curva |r|(t) creciendo frame a frame.
      Fila 1, col 1: curva E(t) creciendo frame a frame.
    cada_cuantos: un frame de animación cada esta cantidad de pasos.
    interval: milisegundos entre frames."""

    # ── Precalcular TODOS los pasos ──────────────────────────
    # Guardamos theta, t, |r|, E en cada paso (para las curvas).
    # Corte temprano si |r| > 0.999 (solo K_n).
    theta_actual = theta.copy()

    # Historiales completos (un valor por paso)
    historial_theta = [theta_actual.copy()]
    historial_t = [0.0]
    historial_r = [abs(parametro_orden(theta_actual))]
    historial_E = [calcular_energia(theta_actual, A, n, tipo_grafo)]

    print(f"\n  Precalculando hasta {pasos} pasos...")
    for k in range(1, pasos + 1):
        theta_actual, _ = paso_euler(theta_actual, A, dt)
        r_k = parametro_orden(theta_actual)
        abs_r_k = abs(r_k)
        E_k = calcular_energia(theta_actual, A, n, tipo_grafo)

        historial_theta.append(theta_actual.copy())
        historial_t.append(k * dt)
        historial_r.append(abs_r_k)
        historial_E.append(E_k)

        # Corte temprano solo para K_n
        if tipo_grafo == 'K' and abs_r_k > 0.999:
            print(f"  Corte temprano en paso {k}: |r| = {abs_r_k:.6f} > 0.999")
            break

    pasos_reales = len(historial_t) - 1  # sin contar t=0
    T_real = historial_t[-1]

    # Elegir los índices de paso que se convierten en frames de animación
    indices_frame = list(range(0, pasos_reales + 1, cada_cuantos))
    if indices_frame[-1] != pasos_reales:
        indices_frame.append(pasos_reales)  # incluir el último

    num_frames = len(indices_frame)
    duracion_estimada = num_frames * interval / 1000.0  # segundos

    print(f"  Pasos calculados: {pasos_reales}")
    print(f"  Frames de animación: {num_frames} (1 cada {cada_cuantos} pasos)")
    print(f"  Duración estimada: {duracion_estimada:.1f} s")

    # Convertir historiales a arrays para slicing rápido
    arr_t = np.array(historial_t)
    arr_r = np.array(historial_r)
    arr_E = np.array(historial_E)
    E_max = max(abs(arr_E.max()), 1.0) * 1.1  # para ylim de E

    # Etiqueta del grafo
    if tipo_grafo == 'K':
        etiq = f"K$_{{{n}}}$"
    elif tipo_grafo == 'D' and k_vecinos is not None:
        etiq = f"C$_{{{n},{k_vecinos}}}$"
    else:
        etiq = f"C$_{{{n}}}$"

    # ── Preparar la figura con gridspec ──────────────────────
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.35, wspace=0.3)

    # Fila 0: círculo (ocupa las 2 columnas)
    ax_circulo = fig.add_subplot(gs[0, :])

    # Fila 1: curvas |r| y E
    ax_r = fig.add_subplot(gs[1, 0])
    ax_E = fig.add_subplot(gs[1, 1])

    fig.suptitle(f"Kuramoto en {etiq}: n={n}, semilla={semilla}",
                 fontsize=14, fontweight='bold')

    # --- Subplot superior: círculo unitario + nodos + flecha r ---
    angulos_circulo = np.linspace(0, 2 * np.pi, 300)
    ax_circulo.plot(np.cos(angulos_circulo), np.sin(angulos_circulo),
                    color='#cccccc', linewidth=1, zorder=1)
    ax_circulo.set_xlim(-1.5, 1.5)
    ax_circulo.set_ylim(-1.5, 1.5)
    ax_circulo.set_aspect('equal')
    ax_circulo.grid(True, alpha=0.3)
    ax_circulo.axhline(0, color='gray', linewidth=0.5)
    ax_circulo.axvline(0, color='gray', linewidth=0.5)

    # Nodos (scatter, se actualiza con set_offsets)
    x0 = np.cos(historial_theta[0])
    y0 = np.sin(historial_theta[0])
    puntos = ax_circulo.scatter(x0, y0, s=120, c='#2196F3',
                                 edgecolors='#0D47A1', linewidths=1.5,
                                 zorder=3)

    # Etiquetas de índice
    etiquetas = []
    for i in range(n):
        txt = ax_circulo.text(x0[i] * 1.15, y0[i] * 1.15, str(i),
                              fontsize=9, ha='center', va='center',
                              fontweight='bold', color='#333333', zorder=4)
        etiquetas.append(txt)

    # Flecha del vector r (quiver desde el origen)
    r0 = parametro_orden(historial_theta[0])
    flecha_r = ax_circulo.quiver(0, 0, r0.real, r0.imag,
                                  angles='xy', scale_units='xy', scale=1,
                                  color='#F44336', width=0.015, zorder=5)
    # Punto gordo en la punta de r
    punto_r, = ax_circulo.plot(r0.real, r0.imag, 'o', color='#F44336',
                                markersize=10, zorder=6)

    titulo_circulo = ax_circulo.set_title(
        f"t = 0.00   |r| = {abs(r0):.3f}", fontsize=12, fontweight='bold'
    )

    # --- Subplot inferior izquierdo: |r|(t) ---
    linea_r, = ax_r.plot([], [], color='#2196F3', linewidth=1.5)
    ax_r.set_xlim(0, T_real * 1.02)
    ax_r.set_ylim(0, 1.05)
    ax_r.set_xlabel("t")
    ax_r.set_ylabel("|r|")
    ax_r.set_title("|r|(t)", fontweight='bold')
    ax_r.grid(True, alpha=0.3)
    ax_r.axhline(1.0, color='gray', linewidth=0.5, linestyle='--')

    # --- Subplot inferior derecho: E(t) ---
    linea_E, = ax_E.plot([], [], color='#F44336', linewidth=1.5)
    ax_E.set_xlim(0, T_real * 1.02)
    ax_E.set_ylim(min(arr_E.min() * 0.9, -0.1), E_max)
    ax_E.set_xlabel("t")
    ax_E.set_ylabel("E")
    ax_E.set_title("E(t)", fontweight='bold')
    ax_E.grid(True, alpha=0.3)

    # ── Función de actualización para cada frame ─────────────
    def actualizar(frame_num):
        """Actualiza nodos, flecha r, y curvas |r|/E hasta el paso actual."""
        # Índice en el historial completo para este frame
        k_actual = indices_frame[frame_num]
        th = historial_theta[k_actual]
        t = arr_t[k_actual]

        # -- Actualizar nodos en el círculo --
        x = np.cos(th)
        y = np.sin(th)
        puntos.set_offsets(np.column_stack([x, y]))

        # Actualizar etiquetas
        for i in range(n):
            etiquetas[i].set_position((x[i] * 1.15, y[i] * 1.15))

        # -- Actualizar flecha r --
        r = parametro_orden(th)
        flecha_r.set_UVC(r.real, r.imag)
        punto_r.set_data([r.real], [r.imag])

        # Título con t y |r|
        texto_titulo = f"t = {t:.2f}   |r| = {abs(r):.3f}"
        if frame_num == num_frames - 1 and tipo_grafo in ('C', 'D') and cond_inicial in ('T', 'X') and abs(r) < 0.5:
            texto_titulo += "  [Se trabó en el twisted state]"
        titulo_circulo.set_text(texto_titulo)

        # -- Actualizar curvas (hasta el paso actual) --
        linea_r.set_data(arr_t[:k_actual + 1], arr_r[:k_actual + 1])
        linea_E.set_data(arr_t[:k_actual + 1], arr_E[:k_actual + 1])

        return puntos, flecha_r, punto_r, linea_r, linea_E

    # ── Crear animación ──────────────────────────────────────
    anim = FuncAnimation(fig, actualizar, frames=num_frames,
                         interval=interval, repeat=False, blit=False)

    theta_final = historial_theta[-1]
    if tipo_grafo in ('C', 'D') and cond_inicial in ('T', 'X') and abs(parametro_orden(theta_final)) < 0.5:
        print("\n  → Se trabó en el twisted state (|r| bajo, no sincroniza globalmente).")

    return theta_final, anim, fig


# ──────────────────────────────────────────────────────────────
# Modo Bosquejo: matrices, espectro, un paso de Euler
# ──────────────────────────────────────────────────────────────

def bosquejo(theta, A, n, semilla, tipo_grafo='K', k_vecinos=None):
    """Muestra matrices, espectro, un paso de Euler y figuras estáticas."""
    if tipo_grafo == 'K':
        etiq = f"K_{n}"
    elif tipo_grafo == 'D' and k_vecinos is not None:
        etiq = f"C_{{{n},{k_vecinos}}}"
    else:
        etiq = f"C_{n}"

    # ── Ángulos iniciales ────────────────────────────────────
    print(f"\n{'='*50}")
    print("  Ángulos iniciales (grados sexagesimales)")
    print(f"{'='*50}")
    grados = rad_a_grados(theta)
    for i in range(n):
        print(f"  theta[{i}] = {grados[i]:8.2f}°")

    # ── Matrices ─────────────────────────────────────────────
    imprimir_matriz(f"Matriz de adyacencia A ({etiq})", A)

    D, L = laplaciano(A)
    imprimir_matriz("Matriz de grados D", D)
    imprimir_matriz("Laplaciano L = D - A", L)

    # ── Velocidades ──────────────────────────────────────────
    dtheta = velocidad(theta, A)
    print(f"\n{'='*50}")
    print("  Velocidades dtheta/dt")
    print(f"{'='*50}")
    for i in range(n):
        print(f"  dtheta[{i}] = {dtheta[i]:+.6f} rad/s  "
              f"({rad_a_grados(dtheta[i]):+.2f}°/s)")

    # ── Parámetro de orden ───────────────────────────────────
    r = parametro_orden(theta)
    print(f"\n{'='*50}")
    print("  Parámetro de orden")
    print(f"{'='*50}")
    print(f"  r       = {r:.6f}")
    print(f"  |r|     = {abs(r):.6f}")
    print(f"  arg(r)  = {np.degrees(np.angle(r)):.2f}°")

    # ── Energía ──────────────────────────────────────────────
    E_antes = calcular_energia(theta, A, n, tipo_grafo)
    formula = "E = (n²/2)(1 - |r|²)" if tipo_grafo == 'K' else "E = Σ (1 - cos(θ_u - θ_v))"
    print(f"\n{'='*50}")
    print(f"  Energía ({formula})")
    print(f"{'='*50}")
    print(f"  E = {E_antes:.6f}")

    # ── Espectro del laplaciano ──────────────────────────────
    autovalores, autovectores = espectro_laplaciano(L)
    imprimir_autovalores(autovalores)

    print(f"\n{'='*50}")
    print("  Autovectores (columnas)")
    print(f"{'='*50}")
    print(autovectores)

    fiedler = autovalores[1]

    # ── Paso de Euler ────────────────────────────────────────
    dt = 0.1
    theta_nuevo, _ = paso_euler(theta, A, dt)
    E_despues = calcular_energia(theta_nuevo, A, n, tipo_grafo)

    print(f"\n{'='*50}")
    print(f"  Paso de Euler (dt = {dt})")
    print(f"{'='*50}")
    grados_nuevo = rad_a_grados(theta_nuevo)
    for i in range(n):
        print(f"  theta_nuevo[{i}] = {grados_nuevo[i]:8.2f}°")

    bajo = E_despues < E_antes
    print(f"\n  E antes  = {E_antes:.6f}")
    print(f"  E después = {E_despues:.6f}")
    print(f"  E bajó: {bajo}")

    # ── Figuras ──────────────────────────────────────────────
    fig1 = graficar_circulo(theta, f"Condición inicial — {etiq}")
    fig2 = graficar_circulo(theta_nuevo,
                            f"Después de 1 paso de Euler (dt={dt}) — {etiq}")

    # ── Resumen final ────────────────────────────────────────
    print(f"\n{'╔'+'═'*50+'╗'}")
    print(f"{'║'}  RESUMEN{' '*42}{'║'}")
    print(f"{'╚'+'═'*50+'╝'}")
    print(f"  Grafo   = {etiq}")
    print(f"  n       = {n}")
    print(f"  semilla = {semilla}")
    print(f"  |r|     = {abs(r):.6f}")
    print(f"  E       = {E_antes:.6f}")
    print(f"  Fiedler = {fiedler:.6f}")
    print(f"  E bajó  = {bajo}")

    plt.show()


# ──────────────────────────────────────────────────────────────
# Caché de espectros para comparación K_n vs C_n
# ──────────────────────────────────────────────────────────────

_cache_espectros = {}


# ──────────────────────────────────────────────────────────────
# Resumen final para Simulación y Animación (extendido)
# ──────────────────────────────────────────────────────────────

def imprimir_resumen_simulacion(theta_inicial, theta_final, A, n, semilla,
                                 fiedler, tipo_grafo, cond_inicial, q=None, k_vecinos=None):
    """Imprime el resumen extendido al terminar Simulación o Animación."""
    r_inicial = parametro_orden(theta_inicial)
    r_final = parametro_orden(theta_final)
    E_inicial = calcular_energia(theta_inicial, A, n, tipo_grafo)
    E_final = calcular_energia(theta_final, A, n, tipo_grafo)
    bajo = E_final < E_inicial
    sinc = abs(r_final) > 0.95

    if tipo_grafo == 'K':
        etiq = f"K_{n}"
    elif tipo_grafo == 'D' and k_vecinos is not None:
        etiq = f"C_{{{n},{k_vecinos}}}"
    else:
        etiq = f"C_{n}"
    if cond_inicial in ('T', 'X') and q is not None:
        etiq += f" (twisted q={q})"

    # Verificación de punto crítico del estado final
    dtheta_final = velocidad(theta_final, A)
    norma_final = np.linalg.norm(dtheta_final)

    # Hessiano del estado final
    H_final = hessiano(theta_final, A)
    mu_H, _, es_estable_H, fiedler_H = analizar_estabilidad(H_final)

    fiedler_signo = 'positivo' if fiedler_H > 1e-8 else (
        'cero' if abs(fiedler_H) <= 1e-8 else 'negativo')

    print(f"\n{'╔'+'═'*55+'╗'}")
    print(f"{'║'}  RESUMEN FINAL{' '*41}{'║'}")
    print(f"{'╚'+'═'*55+'╝'}")
    print(f"  Grafo          = {etiq}")
    print(f"  n              = {n}")
    print(f"  semilla        = {semilla}")
    print(f"  ||dtheta|| fin = {norma_final:.2e} (¿punto crítico? {norma_final < 1e-10})")
    print(f"  Fiedler (L)    = {fiedler:.6f}")
    print(f"  Hessiano: Fiedler = {fiedler_H:.6f} ({fiedler_signo})")

    # Relación H = c·L solo para twisted state en C_n
    if cond_inicial in ('T', 'X') and q is not None and tipo_grafo == 'C':
        c = np.cos(2 * np.pi * q / n)
        print(f"  cos(2πq/n)     = {c:.6f}")

    print(f"  ¿Es mínimo local estable? {'Sí' if es_estable_H else 'No'}")

    nota_r = " (≈ 1 ⇒ sincronía global)" if sinc else " (lejos de 1 ⇒ NO es sincronía)"
    nota_E = " (≈ 0 ⇒ mínimo global)" if sinc else " (lejos de 0 ⇒ NO es el global)"
    print(f"  |r| inicial    = {abs(r_inicial):.6f}")
    print(f"  |r| final      = {abs(r_final):.6f}{nota_r}")
    print(f"  E inicial      = {E_inicial:.6f}")
    print(f"  E final        = {E_final:.6f}{nota_E}")
    print(f"  E bajó         = {bajo}")

    # Conclusión
    if tipo_grafo == 'K' and sinc:
        print(f"\n  Conclusión: K_n sincroniza globalmente ✔")
    elif tipo_grafo in ('C', 'D') and cond_inicial in ('T', 'X') and abs(r_final) < 0.5:
        print(f"\n  Conclusión: {etiq} NO sincroniza globalmente (quedó atrapado en twisted state).")
        print(f"  Comparación: K_n siempre sincroniza (|r|→1); {etiq} con twisted state se traba (|r| bajo).")
    elif not sinc:
        print(f"\n  Conclusión: {etiq} NO sincroniza globalmente (|r| lejos de 1).")
    else:
        print(f"\n  Conclusión: Sincronizó globalmente (|r| > 0.95).")

    # Título para el espectro
    if tipo_grafo == 'K':
        titulo_espectro = f"K_{n} ({'sincronía' if sinc else 'estado final'})"
    elif tipo_grafo == 'D' and k_vecinos is not None:
        if cond_inicial in ('T', 'X') and q is not None:
            titulo_espectro = f"C_{{{n},{k_vecinos}}} (twisted q={q})"
        else:
            titulo_espectro = f"C_{{{n},{k_vecinos}}} (estado final)"
    else:
        if cond_inicial in ('T', 'X') and q is not None:
            titulo_espectro = f"C_{n} (twisted q={q})"
        else:
            titulo_espectro = f"C_{n} (estado final)"

    graficar_espectro(mu_H, titulo_espectro)

    # Guardar en caché para posibilitar comparación
    _cache_espectros[tipo_grafo] = {
        'n': n,
        'q': q if q is not None else 1,
        'autovalores': mu_H
    }

    if 'K' in _cache_espectros and 'C' in _cache_espectros:
        print(f"\n  ℹ Mostrando comparación bonus: K_n vs C_n espectro del Hessiano...")
        graficar_comparacion_espectros(
            _cache_espectros['K']['autovalores'],
            _cache_espectros['C']['autovalores'],
            _cache_espectros['K']['n'],
            _cache_espectros['C']['n'],
            _cache_espectros['C']['q']
        )

    return mu_H


# ──────────────────────────────────────────────────────────────
# Ejercicio 3: Barrido de k en C_{n,k} y cota inferior de μ_c
# ──────────────────────────────────────────────────────────────

def barrido_Cnk(n, q=1, verbose=True):
    """Realiza el barrido de k para un n dado en grafos C_{n,k} con twisted state.
    Calcula el Fiedler del Hessiano (μ_1) para cada k y encuentra el máximo k estable.
    Devuelve (k_max_est, mu_cota)."""
    k_max = (n - 1) // 2
    if k_max < 1:
        if verbose:
            print(f"  Para n={n} no hay k ≥ 1 válido (requiere n ≥ 3).")
        return None, 0.0

    if verbose:
        print(f"\n{'='*68}")
        print(f"  Barrido de k en C_{{{n},k}} (twisted state q={q})")
        print(f"{'='*68}")
        print(f"  {'k':>3}  {'||dθ||':>11}  {'μ_1(H)':>14}  {'Estable':>9}  {'μ(C_{n,k})':>14}")
        print(f"  {'─'*3}  {'─'*11}  {'─'*14}  {'─'*9}  {'─'*14}")

    k_max_est = None
    mu_cota = 0.0
    resultados = []

    for k in range(1, k_max + 1):
        A = crear_grafo_Cnk(n, k)
        theta = np.array([2 * np.pi * q * v / n for v in range(n)])

        # Verificación de equilibrio
        dtheta = velocidad(theta, A)
        norma_dtheta = np.linalg.norm(dtheta)

        # Hessiano y autovalor de Fiedler (modo m=1, el más peligroso)
        H = hessiano(theta, A)
        autovalores = np.linalg.eigh(H)[0]
        mu_1 = autovalores[1]
        es_estable = mu_1 > 1e-8
        mu_conectividad = (2 * k + 1) / n

        resultados.append((k, norma_dtheta, mu_1, es_estable, mu_conectividad))

        if es_estable:
            k_max_est = k
            mu_cota = mu_conectividad

        if verbose:
            est_str = "✔ SÍ" if es_estable else "✘ NO"
            print(f"  {k:3d}  {norma_dtheta:11.2e}  {mu_1:14.6f}  {est_str:>9}  {mu_conectividad:14.4f}")

    if verbose:
        print(f"\n{'─'*68}")
        if k_max_est is not None:
            print(f"  Máximo k estable  : k = {k_max_est}")
            print(f"  Su conectividad   : μ = (2k+1)/n = (2·{k_max_est}+1)/{n} = {mu_cota:.4f}")
            print(f"  Cota inferior μ_c : μ_c > {mu_cota:.4f}")

            if n % 2 != 0 and k_max_est == (n - 1) // 2:
                print(f"  ⚠ Atención: este k da C_{{{n},{k_max_est}}} = K_{n}, y K_n no tiene espurios.")
                print(f"    Revisar si es estable (probablemente NO).")
        else:
            print(f"  No se encontraron estados twisted estables para n={n}, q={q}.")

    return k_max_est, mu_cota


def barrido_varios_n(n_max=15, q=1):
    """Este barrido da la cota inferior de μ_c usando la familia de twisted states en C_{n,k}.
    El techo esperado es ≈0.68 (Yoneda et al. 2021: 0.6838).
    Para superar 11/16 = 0.6875 (Canale) o 0.6929 (S_7), hay que usar otras familias
    (clústeres, no twisted states)."""
    if n_max < 5:
        n_max = 5

    print(f"\n{'='*70}")
    print(f"  Barrido sobre varios n: n = 5 .. {n_max} (twisted q={q})")
    print(f"{'='*70}")
    print(f"  {'n':>4}  {'k_max':>7}  {'μ(C_{n,k_max})':>16}  {'¿Mejora cota?':>16}")
    print(f"  {'─'*4}  {'─'*7}  {'─'*16}  {'─'*16}")

    lista_n = []
    lista_mu = []
    mejor_mu = 0.0
    mejor_n = None
    mejor_k = None

    for n_val in range(5, n_max + 1):
        k_est, mu_c = barrido_Cnk(n_val, q=q, verbose=False)
        lista_n.append(n_val)
        lista_mu.append(mu_c)

        mejora = mu_c > mejor_mu + 1e-9
        if mejora:
            mejor_mu = mu_c
            mejor_n = n_val
            mejor_k = k_est
            marca_mejora = "★ NUEVA MEJOR"
        else:
            marca_mejora = "-"

        k_str = str(k_est) if k_est is not None else "ninguno"
        print(f"  {n_val:4d}  {k_str:>7}  {mu_c:16.4f}  {marca_mejora:>16}")

    print(f"\n{'─'*70}")
    print(f"  Mejor cota inferior (twisted states): μ_c > {mejor_mu:.4f} (en n={mejor_n}, k={mejor_k})")
    print(f"  Techo de los twisted states: μ ≈ 0.68. Para más, hay que salir de los")
    print(f"  circulantes (clústeres: Canale 11/16 = 0.6875, S_7 = 0.6929).")

    # Gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(lista_n, lista_mu, color='#1E88E5', edgecolor='white',
           linewidth=0.6, label='Cota twisted C_{n,k}', zorder=3)
    ax.set_xlabel("Número de osciladores n", fontsize=11)
    ax.set_ylabel("Cota inferior de conectividad μ", fontsize=11)
    ax.set_title(f"Cota inferior de μ_c en grafos C_{{n,k}} (twisted states q={q})",
                 fontsize=13, fontweight='bold')
    ax.set_xticks(lista_n)
    ax.grid(True, alpha=0.3, axis='y')

    # Líneas de referencia teóricas
    ax.axhline(11 / 16, color='#E53935', linestyle='--', linewidth=1.5,
               label=f"Canale: 11/16 = {11/16:.4f}", zorder=4)
    ax.axhline(0.6929, color='#8E24AA', linestyle=':', linewidth=1.5,
               label="S_7 ≈ 0.6929", zorder=4)

    # Anotar el mejor valor alcanzado
    if mejor_n is not None:
        ax.annotate(f"Mejor: {mejor_mu:.4f}\n(n={mejor_n}, k={mejor_k})",
                    xy=(mejor_n, mejor_mu),
                    xytext=(mejor_n, mejor_mu + 0.05),
                    fontsize=9, fontweight='bold', color='#0D47A1', ha='center',
                    arrowprops=dict(arrowstyle='->', color='#0D47A1', lw=1.2),
                    zorder=5)

    ax.set_ylim(0, 0.80)
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()

    return mejor_mu, mejor_n, mejor_k, fig


# ──────────────────────────────────────────────────────────────
# Programa principal
# ──────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║   Modelo de Kuramoto – K_n / C_n / C_{n,k}       ║")
    print("╚══════════════════════════════════════════════════╝")

    # ── Menú: tipo de grafo ──────────────────────────────────
    print("\nElegí tipo de grafo:")
    print("  [K] Completo (K_n)")
    print("  [C] Ciclo (C_n)")
    print("  [D] Denso / Anillo ampliado (C_{n,k})")
    print("  [W] Barrido C_{n,k} (Warm-up 3: cotas de μ_c)")
    print("  [P] Comparación K_n vs C_n (espectro del Hessiano)")
    while True:
        tipo_grafo = input("Opción (K/C/D/W/P): ").strip().upper()
        if tipo_grafo in ('K', 'C', 'D', 'W', 'P'):
            break
        if tipo_grafo == '':
            tipo_grafo = 'K'
            break
        print("  Opción no válida. Elegí K, C, D, W o P.")

    # Opción W: Barrido de C_{n,k}
    if tipo_grafo == 'W':
        print("\n--- Warm-up 3: Barrido de C_{n,k} y cota inferior de μ_c ---")
        print("  [1] Barrido de k para un n dado")
        print("  [2] Barrido sobre varios n (n=5 a n_max)")
        opcion_w = input("Opción (1/2, Enter = 1): ").strip()
        if opcion_w == '2':
            entrada_nm = input("n_max (Enter = 15): ").strip()
            n_max = int(entrada_nm) if entrada_nm else 15
            if n_max < 5:
                n_max = 5
            entrada_qw = input("q (vueltas, Enter = 1): ").strip()
            qw = int(entrada_qw) if entrada_qw else 1
            barrido_varios_n(n_max, qw)
            plt.show()
        else:
            entrada_nw = input("n (Enter = 8, n ≥ 3): ").strip()
            nw = int(entrada_nw) if entrada_nw else 8
            if nw < 3:
                nw = 8
            entrada_qw = input("q (vueltas, Enter = 1): ").strip()
            qw = int(entrada_qw) if entrada_qw else 1
            barrido_Cnk(nw, qw, verbose=True)
        return

    # Opción P: Comparación directa
    if tipo_grafo == 'P':
        print("\n--- Comparación K_n vs C_n (espectro del Hessiano) ---")
        entrada_n = input("Ingresá n (Enter = 5, n ≥ 5): ").strip()
        n_comp = int(entrada_n) if entrada_n else 5
        if n_comp < 3:
            n_comp = 5
        entrada_q = input("Ingresá q para C_n (Enter = 1): ").strip()
        q_comp = int(entrada_q) if entrada_q else 1

        # K_n en sincronía (todos los theta iguales): H = L(K_n)
        A_K = crear_grafo_K(n_comp)
        _, L_K = laplaciano(A_K)
        autov_K, _ = espectro_laplaciano(L_K)

        # C_n en twisted state q: H = cos(2πq/n) * L(C_n)
        theta_C = np.array([2 * np.pi * q_comp * v / n_comp for v in range(n_comp)])
        A_C = crear_grafo_C(n_comp)
        H_C = hessiano(theta_C, A_C)
        autov_C, _ = np.linalg.eigh(H_C)

        print(f"\n  Autovalores K_{n_comp} (sincronía): {np.round(autov_K, 4)}")
        print(f"  Autovalores C_{n_comp} (twisted q={q_comp}): {np.round(autov_C, 4)}")
        c = np.cos(2 * np.pi * q_comp / n_comp)
        print(f"  Factor de escala c = cos(2π·{q_comp}/{n_comp}) = {c:.4f}")

        graficar_comparacion_espectros(autov_K, autov_C, n_comp, n_comp, q_comp)
        plt.show()
        return

    # ── Menú: modo de ejecución ──────────────────────────────
    print("\nElegí modo de ejecución:")
    print("  [B] Bosquejo (matrices, espectro, 1 paso de Euler)")
    print("  [S] Simulación (bucle completo, curvas |r| y E)")
    print("  [A] Animación (nodos moviéndose en el círculo)")
    while True:
        modo = input("Modo (B/S/A): ").strip().upper()
        if modo in ('B', 'S', 'A'):
            break
        print("  Opción no válida. Elegí B, S o A.")

    # ── Pedir n ──────────────────────────────────────────────
    n_min = 3 if tipo_grafo in ('C', 'D') else 2
    while True:
        try:
            n = int(input(f"\nIngresá n (número de nodos, n ≥ {n_min}): "))
            if n >= n_min:
                break
            print(f"  n debe ser ≥ {n_min}.")
        except ValueError:
            print("  Ingresá un número entero.")

    if tipo_grafo == 'C':
        if n < 5:
            print(f"  ℹ  Nota: para el análisis del twisted state, n ≥ 5 es el caso interesante.")
        encontrar_twisted_estables(n)

    # ── Pedir k si el grafo es C_{n,k} ───────────────────────
    k_vecinos = None
    if tipo_grafo == 'D':
        k_max = (n - 1) // 2
        while True:
            try:
                entrada_k = input(f"Ingresá k (vecinos por lado, 1 ≤ k ≤ {k_max}, Enter = {min(2, k_max)}): ").strip()
                k_val = int(entrada_k) if entrada_k else min(2, k_max)
                if 1 <= k_val <= k_max:
                    k_vecinos = k_val
                    break
                if n % 2 == 0 and k_val == n // 2:
                    print(f"  ⚠ Si n={n} es par y k=n/2={n//2}, el nodo opuesto se contaría dos veces. Limitamos k ≤ {k_max}.")
                else:
                    print(f"  k debe satisfacer 1 ≤ k ≤ {k_max}.")
            except ValueError:
                print("  Ingresá un número entero.")

    # ── Pedir semilla ────────────────────────────────────────
    entrada_semilla = input("\nSemilla (Enter = 42): ").strip()
    if entrada_semilla == '':
        semilla = 42
    else:
        semilla = int(entrada_semilla)

    rng = np.random.default_rng(semilla)
    print(f"\n  Semilla: {semilla} (usar la misma para reproducir)")

    # ── Condición inicial ────────────────────────────────────
    q = None
    cond_inicial = 'R'

    if tipo_grafo in ('C', 'D'):
        print("\nCondición inicial:")
        print("  [R] Aleatoria (uniforme en [0, 2π))")
        print("  [T] Twisted state (θ_v = 2πqv/n)")
        print("  [X] Twisted + ruido (rampa + perturbación chica)")
        while True:
            cond_inicial = input("Condición (R/T/X): ").strip().upper()
            if cond_inicial in ('R', 'T', 'X'):
                break
            if cond_inicial == '':
                cond_inicial = 'R'
                break
            print("  Opción no válida.")

        if cond_inicial in ('T', 'X'):
            entrada_q = input("q (número de vueltas, Enter = 1): ").strip()
            q = int(entrada_q) if entrada_q else 1
            if q == 0:
                q = 1
                print("  q=0 no tiene sentido, usando q=1.")

    # Generar theta según la condición elegida
    if cond_inicial == 'R':
        theta = rng.uniform(0, 2 * np.pi, n)
    elif cond_inicial == 'T':
        theta = np.array([2 * np.pi * q * v / n for v in range(n)])
    elif cond_inicial == 'X':
        theta_base = np.array([2 * np.pi * q * v / n for v in range(n)])
        ruido = rng.normal(0, 0.05, n)
        theta = (theta_base + ruido) % (2 * np.pi)

    # ── Crear grafo ──────────────────────────────────────────
    if tipo_grafo == 'K':
        A = crear_grafo_K(n)
    elif tipo_grafo == 'D':
        A = crear_grafo_Cnk(n, k_vecinos)
    else:
        A = crear_grafo_C(n)

    D, L = laplaciano(A)
    autovalores, _ = espectro_laplaciano(L)
    fiedler = autovalores[1]

    # ── Análisis del twisted state (si corresponde) ──────────
    if cond_inicial in ('T', 'X') and tipo_grafo in ('C', 'D'):
        if cond_inicial == 'T':
            verificar_punto_critico(theta, A)

        H = hessiano(theta, A)
        imprimir_matriz("Hessiano H (Laplaciano con pesos cos)", H)
        mu_H, _, es_estable, fiedler_H = analizar_estabilidad(H)

        if tipo_grafo == 'C':
            verificar_relacion_twisted(H, L, n, q)

    # ── Despachar según el modo ──────────────────────────────
    if modo == 'B':
        bosquejo(theta, A, n, semilla, tipo_grafo, k_vecinos)

    elif modo == 'S':
        dt, T, pasos = pedir_dt_y_T(autovalores)
        theta_final, tiempos, r_abs_hist, E_hist = simular(
            theta, A, n, dt, pasos, tipo_grafo, k_vecinos
        )
        imprimir_resumen_simulacion(
            theta, theta_final, A, n, semilla, fiedler,
            tipo_grafo, cond_inicial, q, k_vecinos
        )
        plt.show()

    elif modo == 'A':
        dt, T, pasos = pedir_dt_y_T(autovalores)
        entrada_cc = input("cada_cuantos (1 frame cada X pasos, Enter = 1): ").strip()
        cada_cuantos = int(entrada_cc) if entrada_cc else 1
        if cada_cuantos < 1:
            cada_cuantos = 1

        entrada_iv = input("interval en ms (Enter = 100): ").strip()
        interval = int(entrada_iv) if entrada_iv else 100
        if interval < 1:
            interval = 1

        theta_final, anim, fig = animar(
            theta, A, n, dt, pasos, semilla, tipo_grafo,
            cada_cuantos, interval, cond_inicial, q, k_vecinos
        )
        imprimir_resumen_simulacion(
            theta, theta_final, A, n, semilla, fiedler,
            tipo_grafo, cond_inicial, q, k_vecinos
        )
        plt.show()


if __name__ == "__main__":
    main()
