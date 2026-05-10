# tfg-competencia-entre-subastas

Este repositorio contiene el código utilizado para el Trabajo de Fin de Grado de Economía **“Competencia entre subastas”**.

El objetivo del modelo es estudiar el comportamiento estratégico de los licitadores cuando no existe una única subasta aislada, sino varias subastas simultáneas de primer precio con objetos idénticos. En este entorno, los jugadores deben decidir cuánto pujar y en qué subasta participar.

El código simula un modelo dinámico mediante Monte Carlo y calcula la mejor respuesta de un jugador frente a una estrategia común de sus rivales, con el fin de analizar la posible existencia de un equilibrio simétrico.

## Descripción del modelo

El modelo considera:

- `N` jugadores.
- `K` objetos idénticos, con `N > K`.
- Jugadores con demanda unitaria, es decir, cada jugador puede obtener como máximo un objeto.
- Subastas dinámicas de primer precio.
- Valoraciones privadas en el intervalo `[0, 1]`.
- Una estrategia de puja resumida mediante un parámetro de agresividad `lambda`.

La puja del jugador se calcula como:

```python
bid = lambda * valoracion + (1 - lambda) * precio_actual