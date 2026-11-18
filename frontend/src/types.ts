export type AgentStatus = 'STOPPED' | 'RUNNING' | 'PAUSED' | 'HALTED'

export interface Indicators {
  sma20: number | null
  dist_sma20: number | null
  vol20: number | null
  rsi14: number | null
}

export interface Decision {
  id: number
  run_id: string
  candle_time: string
  close: number
  forecast: number | null
  indicators: string
  proposed_action: string
  justification: string | null
  final_action: string
  rejected: number
  reject_reason: string | null
  llm_latency_ms: number | null
  tool_calls: string
  created_at: string
}

export interface Portfolio {
  run_id: string
  cash: number
  qty: number
  avg_price: number
  fees: number
  slippage: number
  initial_cash: number
  price: number | null
  equity: number
  exposure: number
}

export interface RiskConfig {
  initial_capital: number
  fee_rate: number
  slippage_bps: number
  max_exposure_pct: number
  max_daily_loss_pct: number
  max_data_age_hours: number
  min_notional: number
  qty_step: number
  signal_threshold: number
}

export interface Status {
  version: string
  symbol: string
  interval: string
  mode: 'PAPER'
  live_trading_enabled: boolean
  agent: { status: AgentStatus; reason: string | null; since: string | null; cycle_seconds: number }
  data: {
    candle_time: string
    close: number
    forecast: number
    indicators: Indicators
    data_age_hours: number
  } | null
  last_decision: Decision | null
  portfolio: Portfolio
  risk: RiskConfig
  model: { version: string; type: string; hidden: number; device_entrenamiento: string }
  llm: { model: string; url: string }
  replay_progress: { tag: string; done: number; total: number; candle: string } | null
}

export interface EquityRow {
  candle_time: string
  equity: number
  price: number
  cash: number
  qty: number
}

export interface EventRow {
  id: number
  ts: string
  run_id: string | null
  level: 'info' | 'warning' | 'error'
  kind: string
  message: string
}

export interface RunMetrics {
  run_id: string
  ciclos: number
  desde?: string
  hasta?: string
  rendimiento_neto?: number
  max_drawdown?: number
  compras?: number
  ventas?: number
  win_rate?: number | null
  costos_totales?: number
  acciones_rechazadas?: number
  latencia_llm_ms_media?: number | null
}

export interface Run {
  id: string
  mode: 'live' | 'replay'
  strategy: 'agent_llm' | 'model_only' | 'buy_hold'
  created_at: string
  status: string
  metrics: RunMetrics
}

export interface LlmHealth {
  ollama_version: string
  model: string
  descargado: boolean
  en_memoria: boolean
  local: boolean
  detalles: { parameter_size: string; quantization_level: string } | null
  hardware: { cpu: string; ram_gb: number; sistema: string }
}
