# UI · Vue 3 + Vite + TypeScript

Interfaz del MEXC Paper Trading Agent. Instrucciones completas en el [README principal](../README.md).

```bash
npm install
npm run dev        # http://127.0.0.1:5173 (proxy /api → backend en 127.0.0.1:8000)
npm run build      # vue-tsc (verificación de tipos) + build de producción
```

La API key **no** se compila en el bundle ni usa variables `VITE_`: el usuario la escribe al entrar y se guarda solo en `sessionStorage`.
