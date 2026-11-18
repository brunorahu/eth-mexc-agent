// Cliente de la API del backend. La API key NO está en el código ni en variables VITE_:
// la persona la escribe al abrir la UI y se guarda solo en sessionStorage de su navegador.
import type { Decision, EquityRow, EventRow, LlmHealth, RiskConfig, Run, Status } from './types'

const KEY_STORAGE = 'mexc-agent-api-key'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export const apiKey = {
  get: (): string => sessionStorage.getItem(KEY_STORAGE) ?? '',
  set: (k: string) => sessionStorage.setItem(KEY_STORAGE, k),
  clear: () => sessionStorage.removeItem(KEY_STORAGE),
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey.get(), ...(init.headers ?? {}) },
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* respuesta sin JSON: se conserva statusText */
    }
    throw new ApiError(res.status, detail || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) })

export const api = {
  health: () => fetch('/api/health').then((r) => r.json()),
  status: () => request<Status>('/status'),
  start: () => post<{ status: string }>('/agent/start'),
  pause: () => post<{ status: string }>('/agent/pause'),
  stop: () => post<{ status: string }>('/agent/stop'),
  cycle: () => post<{ ok: boolean; message?: string; result?: Decision }>('/agent/cycle'),
  liquidate: () => post<{ ok: boolean; message?: string }>('/agent/liquidate'),
  decisions: (runId = 'live', limit = 30) => request<Decision[]>(`/decisions?run_id=${encodeURIComponent(runId)}&limit=${limit}`),
  equity: (runId = 'live') => request<EquityRow[]>(`/equity?run_id=${encodeURIComponent(runId)}`),
  events: (limit = 40) => request<EventRow[]>(`/events?limit=${limit}`),
  runs: () => request<Run[]>('/runs'),
  replay: (body: { start: string; end: string; limit: number; strategies: string[] }) =>
    post<{ tag: string; runs: Record<string, string>; candles: number }>('/replay', body),
  risk: () => request<RiskConfig>('/config/risk'),
  updateRisk: (changes: Partial<RiskConfig>) => request<RiskConfig>('/config/risk', { method: 'PUT', body: JSON.stringify(changes) }),
  llmHealth: () => request<LlmHealth>('/llm/health'),
}
