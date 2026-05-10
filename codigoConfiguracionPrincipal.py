"""
Trabajo de Fin de Grado de Economía - COMPETENCIA ENTRE SUBASTAS
Autora: Vera del Carmen Burgos Galán
Version: 28 de abril de 2026
"""

import numpy as np


# Construye el vector de valoraciones en el caso condicionado.
# El postor 1 tiene valoración v1, N-K rivales quedan por debajo de v1 y K-1 rivales quedan por encima. 
# La permutación final evita fijar posiciones concretas para los rivales.
def build_conditional_valuations(
    v1: float,
    u_low: np.ndarray,
    u_high: np.ndarray,
    perm: np.ndarray,
) -> np.ndarray:
    rivals_low = v1 * u_low
    rivals_high = v1 + (1.0 - v1) * u_high
    rivals = np.concatenate((rivals_low, rivals_high))
    rivals = rivals[perm]
    return np.concatenate(([v1], rivals))


# Simula una trayectoria completa de la subasta dinámica para unas valoraciones dadas y unas intensidades de puja fijadas.
def simulate_one_auction(
    vals: np.ndarray,
    N: int,
    K: int,
    T: int,
    lam1: float,
    lam_rivals: float,
    rng: np.random.Generator,
) -> tuple[float, int]:
    #Al principio, los precios son cero y ningún objeto tiene ganador provisional.
    prices = np.zeros(K)
    winners = np.full(K, -1, dtype=int)
    
    #Durante T periodos, se sortea aleatoriamente un orden de jugadores
    for _ in range(T):
        order = rng.permutation(N)
        
        #Dentro cada periodo, solo actúan los primeros K jugadores de ese orden. 
        for r in range(K):
            i = int(order[r])

            # Si el postor i ya va ganando algún objeto, no vuelve a pujar.
            if i in winners:
                continue

            # Los objetos elegibles son aquellos cuyo precio actual es menor que la valoración del postor i.
            eligible = np.where(prices < vals[i])[0]
            if eligible.size == 0:
                continue

            # Regla elegir el objeto más barato 
            # (dentro de los objetos que todavía son rentables para ese jugador, selecciona el que tiene menor precio vigente)
            l = int(eligible[np.argmin(prices[eligible])])

            #Cálculo de la puja
            lam = lam1 if i == 0 else lam_rivals
            bid = lam * vals[i] + (1.0 - lam) * prices[l]

            # Solo se actualiza si la nueva puja mejora el precio actual.
            if bid > prices[l]:
                prices[l] = bid
                winners[l] = i
    
    # Cálculo del resultado final del jugador 1.
    # Se revisan K objetos para comprobar si jugador 1 aparece como ganador provisional.
    # Si gana, su payoff = v1 - precio final que él mismo paga por el objeto.
    # Si no gana, su payoff = 0.  
    payoff = 0.0
    win = 0
    v1 = float(vals[0])

    for l in range(K):
        if winners[l] == 0:
            win = 1
            payoff = v1 - prices[l]
            break

    return payoff, win


# Para cada valor de lambda_rivals, selecciona el valor de lambda1 que maximiza el pago esperado del postor 1 en la matriz de pagos.
# Devuelve el vector de mejores respuestas y los pagos asociados.
def best_response(grid: np.ndarray, payoff_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argmax(payoff_matrix, axis=1)
    br_lambda = grid[idx]
    br_payoff = payoff_matrix[np.arange(grid.size), idx]
    return br_lambda, br_payoff


# Busca candidatos a equilibrio simétrico comparando la función de mejor respuesta con la diagonal lambda1 = lambda_rivals.
# Identifica puntos fijos exactos, puntos cercanos e intervalos de cruce.
def find_symmetric_equilibrium(grid: np.ndarray, payoff_matrix: np.ndarray) -> dict[str, object]:
    br_lambda, _ = best_response(grid, payoff_matrix)
    diff = br_lambda - grid   # Diferencia entre la mejor respuesta y la estrategia de los rivales.
    step = grid[1] - grid[0]    
    tol = step / 2            # Distancia entre puntos consecutivos en la malla.

    fixed_idx = np.where(np.abs(diff) <= tol)[0]    # Identifica los puntos de la malla en los que la mejor respuesta coincide aproximadamente con la diagonal.

    min_abs_diff = np.min(np.abs(diff))
    near_idx = np.where(np.abs(diff) == min_abs_diff)[0]    # Busca punto más cercano al equilibrio.

    # Si diff cambia de signo entre dos puntos consecutivos de la malla, la función de mejor respuesta pasa a estar por debajo, o al revés.
    # Ese cambio de signo identifica un intervalo candidato a contener un equilibrio simétrico interior.
    crossing_intervals = []
    for i in range(len(grid) - 1):
        # Si la diferencia es exactamente cero, la mejor respuesta coincide con la estrategia rival en ese mismo valor y aparece un punto fijo exacto.
        if diff[i] == 0:
            crossing_intervals.append((float(grid[i]), float(grid[i])))
        # Si el producto es negativo, diff[i] y diff[i+1] tienen signos opuestos, se guarda el intervalo entre ambos puntos.
        elif diff[i] * diff[i + 1] < 0:
            crossing_intervals.append((float(grid[i]), float(grid[i + 1])))

    # Se devuelve la información: puntos fijos exactos, puntos mas cercanos al equilibrio, intervalos de cruce y función de mejor respuesta evaluada en la malla.
    return {
        "fixed_points_idx": fixed_idx.tolist(),
        "fixed_points_lambda": grid[fixed_idx].tolist(),
        "near_fixed_points_idx": near_idx.tolist(),
        "near_fixed_points_lambda": grid[near_idx].tolist(),
        "min_abs_diff": float(min_abs_diff),
        "crossing_intervals": crossing_intervals,
        "br_lambda": br_lambda.tolist(),
        "grid": grid.tolist(),
    }


# Recorre toda la malla de estrategias posibles.
# Para cada pareja (lambda_rivals, lambda1), repite MC simulaciones y calcula el pago esperado del jugador 1 y su probabilidad de ganar. 
def run_grid_study(
    N: int = 5,
    K: int = 3,
    T: int = 20,
    MC: int = 1000,
    grid_size: int = 21,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (N > K >= 1):
        raise ValueError("Se requiere que N > K >= 1.")
    if T <= 0 or MC <= 0 or grid_size < 2:
        raise ValueError("Se requiere que T > 0, MC > 0 y grid_size >= 2.")

    # Generador maestro a partir del cual se construyen todas las realizaciones aleatorias.
    master_rng = np.random.default_rng(seed)

    # Malla uniforme de valores posibles de lambda en el intervalo [0, 1].
    grid = np.linspace(0.0, 1.0, grid_size)

    # Cada fila corresponde a un valor de lambda_rivals y cada columna a un valor de lambda1.
    payoff_matrix = np.zeros((grid_size, grid_size))
    win_matrix = np.zeros((grid_size, grid_size))

    # Estas primitivas aleatorias se generan una sola vez y luego se reutilizan en todas las celdas del grid (common random numbers).
    # realizaciones subyacentes de valores y azar.
    v1_draws = master_rng.uniform(0.0, 1.0, size=MC)
    u_low_draws = master_rng.uniform(0.0, 1.0, size=(MC, N - K))
    u_high_draws = master_rng.uniform(0.0, 1.0, size=(MC, K - 1))
    rival_perms = np.array([master_rng.permutation(N - 1) for _ in range(MC)], dtype=int)
    sim_seeds = master_rng.integers(0, 2**32 - 1, size=MC, dtype=np.uint32)

    # Bucle exterior: fija la agresividad común de los rivales.
    for ir, lam_rivals in enumerate(grid):
        # Bucle intermedio: fija la agresividad del jugador 1.
        for i1, lam1 in enumerate(grid):
            payoff_sum = 0.0
            wins = 0

            # Bucle interior: repite MC trayectorias independientes de la subasta para la pareja concreta de estrategias.
            for m in range(MC):
                # Se construye el vector de valoraciones condicionado para esta réplica.
                vals = build_conditional_valuations(
                    v1=float(v1_draws[m]),
                    u_low=u_low_draws[m],
                    u_high=u_high_draws[m],
                    perm=rival_perms[m],
                )

                # Cada réplica utiliza su propia semilla para generar el orden aleatorio de movimiento dentro de la subasta.
                rng_m = np.random.default_rng(int(sim_seeds[m]))

                payoff, win = simulate_one_auction(
                    vals, N, K, T, float(lam1), float(lam_rivals), rng_m
                )

                # Se acumulan los resultados para luego calcular medias Monte Carlo en esta celda de la matriz.
                payoff_sum += payoff
                wins += win

            # Al terminar las MC réplicas se guardan los promedios de la celda:
            # pago esperado del jugador 1 y frecuencia estimada de victoria.
            payoff_matrix[ir, i1] = payoff_sum / MC
            win_matrix[ir, i1] = wins / MC

    # La función devuelve la malla y las dos matrices resumen sobre las que después se calculan la mejor respuesta y el equilibrio.
    return grid, payoff_matrix, win_matrix


if __name__ == "__main__":

    # Configuración base utilizada en el análisis principal del trabajo.
    config = {
        "N": 7,
        "K": 3,
        "T": 20,
        "MC": 10000,
        "grid_size": 41,
        "seed": 42,
    }

    # Se ejecuta el barrido completo de estrategias sobre la malla.
    grid, payoff_matrix, win_matrix = run_grid_study(**config)

    # A partir de la matriz de pagos se calcula la función de mejor respuesta del jugador 1 y los posibles equilibrios simétricos.
    br_lambda, br_payoff = best_response(grid, payoff_matrix)
    sym_eq = find_symmetric_equilibrium(grid, payoff_matrix)

    # Para cada valor de lambda_rivals se localiza el indice de la mejor respuesta y se recupera la probabilidad de ganar en esa misma celda.
    br_idx = np.argmax(payoff_matrix, axis=1)
    win_on_br = win_matrix[np.arange(grid.size), br_idx]

    # Se imprimen los principales resultados numéricos.
    print("Mejor respuesta de lambda1:", br_lambda)
    print("Pago asociado a la mejor respuesta:", br_payoff)
    print("Probabilidades de ganar sobre la mejor respuesta:", win_on_br)
    print("Candidatos a equilibrio simétrico:", sym_eq)
