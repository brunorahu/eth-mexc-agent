# Evaluación: modelo sin LLM vs agente con LLM vs comprar y mantener

- Modo: **REPLAY** con velas históricas de MEXC (no es conexión en vivo).
- Periodo: 2025-07-21 → 2026-11-15 (482 ciclos de decisión por estrategia); periodo de prueba independiente (TE) del modelo `ethusdt-gru-1.0.0`.
- Ejecución: decisión al cierre de t, ejecución a la apertura de t+1, valoración al cierre de t+1.
- Costos: comisión 0.00% y deslizamiento 5.0 pb por operación; capital inicial 50,000 USDT; liquidación final al último cierre con costos.
- LLM: `qwen2.5:3b` local (Ollama 0.40.16), duración total 18.3 min.

| Estrategia | Rend. neto | Máx. drawdown | Operaciones (compras/ventas) | Win Rate | Costos (USDT) | Rechazadas | Latencia LLM media (ms) |
|---|---|---|---|---|---|---|---|
| Agente con LLM local | -52.89% | -69.33% | 44/44 | +47.73% | 1,447.98 | 0 | 2265 |
| Modelo sin LLM (regla δ) | -52.89% | -69.33% | 44/44 | +47.73% | 1,447.98 | 0 | — |
| Comprar y mantener | -49.06% | -64.89% | 1/0 | no definido | 23.74 | 0 | — |
