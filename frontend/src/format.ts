export const pct = (x: number | null | undefined, d = 2) =>
  x === null || x === undefined || Number.isNaN(x) ? '—' : `${x >= 0 ? '+' : ''}${(x * 100).toFixed(d)}%`

export const usd = (x: number | null | undefined, d = 2) =>
  x === null || x === undefined ? '—' : x.toLocaleString('es-MX', { minimumFractionDigits: d, maximumFractionDigits: d })

export const num = (x: number | null | undefined, d = 2) => (x === null || x === undefined ? '—' : x.toFixed(d))

export const time = (iso: string | null | undefined) =>
  iso ? new Date(iso).toLocaleString('es-MX', { dateStyle: 'short', timeStyle: 'short' }) : '—'

export const actionLabel: Record<string, string> = {
  BUY: 'Comprar',
  SELL: 'Vender',
  HOLD: 'Mantener / efectivo',
  INVALID: 'Inválida',
  LIQUIDATE: 'Liquidar',
}

export const statusLabel: Record<string, string> = {
  RUNNING: 'En ejecución',
  PAUSED: 'En pausa',
  STOPPED: 'Detenido',
  HALTED: 'Bloqueado por riesgo',
}
