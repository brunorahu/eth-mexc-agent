## Evaluación (ETHUSDT · Modelo GRU)

**Replay** con velas históricas diarias de MEXC sobre el periodo de prueba independiente (TE) del modelo `ethusdt-gru-1.0.0`: **2025-07-21 → 2026-11-15 (482 ciclos por estrategia)**, con capital inicial de 50,000 USDT, comisión 0.0% y deslizamiento de 5 pb por operación.

| Estrategia | Rend. neto | Máx. drawdown | Operaciones (compras/ventas) | Win Rate | Costos (USDT) | Rechazadas | Latencia LLM media (ms) |
|---|---|---|---|---|---|---|---|
| Agente con LLM local (`qwen2.5:3b`) | **−52.89%** | −69.33% | 44 / 44 | 47.73% | 1,447.98 | 0 | 2,265 |
| Modelo sin LLM (regla δ) | **−52.89%** | −69.33% | 44 / 44 | 47.73% | 1,447.98 | 0 | — |
| Comprar y mantener | **−49.06%** | −64.89% | 1 / 0 | no definido | 23.74 | 0 | — |

**Interpretación de resultados:**
- **Contexto de mercado y costos de fricción:** Durante el periodo de prueba independiente (TE), ETHUSDT experimentó una tendencia bajista pronunciada (−49.06% en *Buy & Hold*). Aunque el modelo GRU alcanzó un *Win Rate* del 47.73% en 44 operaciones cerradas, la alta frecuencia de rotación generó 1,447.98 USDT en costos por deslizamiento, situando el rendimiento neto en −52.89%.
- **Aporte del LLM local:** El agente con `qwen2.5:3b` completó los 482 ciclos con **0 rechazos** por parte del motor de riesgo y una latencia media de 2.27 s por decisión, replicando las señales del modelo GRU y traduciéndolas a justificaciones auditable en español, sin aportar rentabilidad adicional al no contar con datos fundamentales o de noticias externas.