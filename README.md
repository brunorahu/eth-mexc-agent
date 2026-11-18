## Versión y tecnologías en uso

- **Versión actual del BOT:** `v1.0.0` (Modelo `ethusdt-gru-1.0.0`, activo `ETHUSDT`, temporalidad `1d`)
- **Estado del repositorio:** Público para revisión académica

| Componente | Tecnología | Versión / Detalle |
|---|---|---|
| Aplicación / Bot | MEXC Paper Trading Agent (`ETHUSDT`) | **v1.0.0** |
| Modelo predictivo | Red Neuronal Recurrente **GRU** (`h=32`) | PyTorch 2.15.0 (entrenado en Google Colab GPU Tesla T4) |
| LLM local | Ollama + `qwen2.5:3b` (3.1B, Q4_K_M) | Ollama 0.40.16 (inferencia local en `127.0.0.1:11434`) |
| Backend | Python + FastAPI + Uvicorn | Python 3.14.4 · FastAPI 0.163.2 · Uvicorn 0.54.0 |
| Datos y ML | pandas · NumPy · scikit-learn · joblib | 3.1.1 · 2.5.5 · 1.9.1 · 1.6.0 |
| Validación y HTTP | httpx · Pydantic | 0.28.1 · 2.14.3 |
| Persistencia | SQLite (modo WAL) | Incluida en Python 3.14 |
| Interfaz Web (UI) | Vue 3 + Vite + TypeScript | Vue 3.5 · Vite 8.3 · TypeScript 6.0 |
| API de mercado | MEXC Spot API v3 (endpoints públicos) | `klines`, `ticker/price`, `exchangeInfo` |
| Pruebas | pytest | 9.2.0 (27 pruebas automatizadas) |
| Entorno de ejecución | Windows 11 (AMD Ryzen AI 9 HX 370, 16 GB RAM) | Ejecución local |

---

## Historial de commits y pushes

| Fecha (UTC-6) | Versión | Commit / Push | Resumen de cambios relevantes |
|---|---|---|---|
| 2026-11-17 | `v1.0.0` | `feat: implementar agente de paper trading ETHUSDT con modelo GRU y LLM local` | Implementación completa del bot en modo Paper para `ETHUSDT` (`1d`): cliente MEXC v3 con reintentos, modelo `GRU (h=32)` entrenado en GPU T4, agente con Ollama (`qwen2.5:3b`) y herramientas de solo lectura, motor de riesgo determinista, base de datos SQLite, frontend Vue 3, 27 pruebas `pytest` y evaluación comparativa en TE (482 ciclos). |
| 2026-11-17 | `v1.0.0` | `docs: actualizar README con bitácora de commits, versión del bot y tecnologías` | Documentación de la versión `v1.0.0`, stack tecnológico en Windows 11, tabla de resultados de evaluación (`ETHUSDT`) y registro de commits/pushes según rúbrica. |