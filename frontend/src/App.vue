<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { api, apiKey, ApiError } from './api'
import type { Decision, EquityRow, EventRow, LlmHealth, Run, Status } from './types'
import { actionLabel, num, pct, statusLabel, time, usd } from './format'
import KeyGate from './components/KeyGate.vue'
import EquityChart from './components/EquityChart.vue'
import DecisionsTable from './components/DecisionsTable.vue'
import ReplayPanel from './components/ReplayPanel.vue'
import RiskPanel from './components/RiskPanel.vue'

const authed = ref(false)
const status = ref<Status | null>(null)
const decisions = ref<Decision[]>([])
const equity = ref<EquityRow[]>([])
const events = ref<EventRow[]>([])
const runs = ref<Run[]>([])
const llm = ref<LlmHealth | null>(null)
const llmError = ref('')
const message = ref<{ kind: 'ok' | 'error'; text: string } | null>(null)
const busy = ref(false)
let timer: number | undefined

const s = computed(() => status.value)
const agentStatus = computed(() => s.value?.agent.status ?? 'STOPPED')
const last = computed(() => s.value?.last_decision ?? null)

function notify(kind: 'ok' | 'error', text: string) {
  message.value = { kind, text }
  window.setTimeout(() => (message.value = null), 6000)
}

async function refresh() {
  try {
    const [st, dec, eq, ev, rn] = await Promise.all([api.status(), api.decisions(), api.equity(), api.events(), api.runs()])
    status.value = st
    decisions.value = dec
    equity.value = eq
    events.value = ev
    runs.value = rn
  } catch (e) {
    if (e instanceof ApiError && e.status === 401) logout()
    else notify('error', `Backend sin respuesta: ${(e as Error).message}`)
  }
}

async function refreshLlm() {
  try {
    llm.value = await api.llmHealth()
    llmError.value = ''
  } catch (e) {
    llm.value = null
    llmError.value = (e as Error).message
  }
}

async function action(fn: () => Promise<unknown>, ok: string) {
  busy.value = true
  try {
    const r = (await fn()) as { message?: string; ok?: boolean }
    notify(r && r.ok === false ? 'error' : 'ok', r?.message ?? ok)
  } catch (e) {
    notify('error', (e as Error).message)
  } finally {
    busy.value = false
    await refresh()
  }
}

function logout() {
  apiKey.clear()
  authed.value = false
  window.clearInterval(timer)
}

function onAuthed() {
  authed.value = true
  refresh()
  refreshLlm()
  timer = window.setInterval(refresh, 5000)
}

onMounted(async () => {
  if (apiKey.get()) {
    try {
      await api.status()
      onAuthed()
    } catch {
      apiKey.clear()
    }
  }
})
onUnmounted(() => window.clearInterval(timer))
</script>

<template>
  <KeyGate v-if="!authed" @ok="onAuthed" />

  <div v-else-if="s" class="page">
    <header>
      <div>
        <h1>{{ s.symbol }} · Agente Paper</h1>
        <div class="small muted">
          v{{ s.version }} · modelo {{ s.model.version }} ({{ s.model.type }} h={{ s.model.hidden }}) · LLM local {{ s.llm.model }}
        </div>
      </div>
      <div class="head-right">
        <span class="chip paper">PAPER · sin órdenes reales</span>
        <span :class="['chip', 'st', `st-${agentStatus}`]">{{ statusLabel[agentStatus] }}</span>
        <button class="small" @click="logout">Salir</button>
      </div>
    </header>

    <p v-if="message" :class="['banner', message.kind]">{{ message.text }}</p>
    <p v-if="s.agent.reason" class="small muted reason">Estado: {{ s.agent.reason }} ({{ time(s.agent.since) }})</p>

    <section class="controls panel">
      <button class="primary" :disabled="busy || agentStatus === 'RUNNING'" @click="action(api.start, 'Agente iniciado')">Iniciar</button>
      <button :disabled="busy || agentStatus !== 'RUNNING'" @click="action(api.pause, 'Agente en pausa')">Pausar</button>
      <button class="danger" :disabled="busy || agentStatus === 'STOPPED'" @click="action(api.stop, 'Agente detenido: órdenes bloqueadas')">Detener</button>
      <span class="sep" />
      <button :disabled="busy" @click="action(api.cycle, 'Ciclo ejecutado')">{{ busy ? 'Procesando…' : 'Ejecutar ciclo ahora' }}</button>
      <button class="danger" :disabled="busy || s.portfolio.qty <= 0" @click="action(api.liquidate, 'Posición liquidada')">Liquidar posición</button>
      <span class="small muted">El ciclo automático revisa cada {{ s.agent.cycle_seconds }} s si cerró una vela diaria nueva. Detener no liquida.</span>
    </section>

    <section class="cards">
      <div class="panel card">
        <div class="label">Datos</div>
        <div class="big">{{ s.data ? usd(s.data.close) : '—' }}</div>
        <div class="small muted">Vela {{ s.data?.candle_time ?? '—' }} ({{ s.interval }}) · antigüedad {{ s.data ? num(s.data.data_age_hours, 1) : '—' }} h</div>
      </div>
      <div class="panel card">
        <div class="label">Pronóstico siguiente vela</div>
        <div class="big">{{ pct(s.data?.forecast, 3) }}</div>
        <div class="small muted">Umbral δ {{ pct(s.risk.signal_threshold, 3) }} · señal {{ s.data && s.data.forecast > s.risk.signal_threshold ? 'comprar/mantener' : 'efectivo' }}</div>
      </div>
      <div class="panel card">
        <div class="label">Indicadores</div>
        <div class="small">Distancia SMA20: <b>{{ pct(s.data?.indicators.dist_sma20) }}</b></div>
        <div class="small">Volatilidad 20: <b>{{ pct(s.data?.indicators.vol20) }}</b></div>
        <div class="small">RSI 14: <b>{{ num(s.data?.indicators.rsi14, 1) }}</b></div>
      </div>
      <div class="panel card">
        <div class="label">Acción propuesta</div>
        <div class="big" :class="last ? `act-${last.final_action}` : ''">{{ last ? actionLabel[last.final_action] : '—' }}</div>
        <div class="small muted">LLM: {{ last ? actionLabel[last.proposed_action] ?? last.proposed_action : '—' }}</div>
      </div>
      <div class="panel card">
        <div class="label">Posición</div>
        <div class="big">{{ num(s.portfolio.qty, 6) }} ETH</div>
        <div class="small muted">Efectivo {{ usd(s.portfolio.cash) }} USDT · exposición {{ pct(s.portfolio.exposure, 1) }}</div>
      </div>
      <div class="panel card">
        <div class="label">Patrimonio</div>
        <div class="big">{{ usd(s.portfolio.equity) }}</div>
        <div class="small muted">{{ pct(s.portfolio.equity / s.portfolio.initial_cash - 1) }} vs {{ usd(s.portfolio.initial_cash, 0) }} USDT</div>
      </div>
    </section>

    <section v-if="last" class="panel">
      <h2>Motivo de la última decisión <span class="small muted">· vela {{ last.candle_time }} · {{ time(last.created_at) }}</span></h2>
      <p>{{ last.justification ?? '—' }}</p>
      <p v-if="last.rejected" class="lvl-warning small">Controles de riesgo: {{ last.reject_reason }}</p>
      <p class="small muted">Herramientas consultadas: {{ JSON.parse(last.tool_calls || '[]').join(', ') || 'ninguna' }} · latencia LLM {{ last.llm_latency_ms ? Math.round(last.llm_latency_ms) + ' ms' : '—' }}</p>
    </section>

    <div class="two">
      <section class="panel">
        <h2>Patrimonio (modo en vivo)</h2>
        <EquityChart :rows="equity" :initial="s.portfolio.initial_cash" />
      </section>
      <section class="panel">
        <h2>LLM local</h2>
        <div v-if="llm" class="small">
          <div>Ollama {{ llm.ollama_version }} · {{ llm.model }} ({{ llm.detalles?.parameter_size }}, {{ llm.detalles?.quantization_level }})</div>
          <div>Hardware: {{ llm.hardware.cpu }} · {{ llm.hardware.ram_gb }} GB · {{ llm.hardware.sistema }}</div>
          <div>Inferencia local: <b :class="llm.local ? 'act-BUY' : 'lvl-error'">{{ llm.local ? 'sí (127.0.0.1)' : 'no' }}</b> · en memoria: {{ llm.en_memoria ? 'sí' : 'no' }}</div>
        </div>
        <div v-else class="small lvl-error">Ollama no disponible: {{ llmError }}</div>
        <button class="small" @click="refreshLlm">Verificar</button>
      </section>
    </div>

    <section class="panel">
      <h2>Decisiones (modo en vivo)</h2>
      <DecisionsTable :rows="decisions" />
    </section>

    <section class="panel">
      <h2>Evaluación con replay histórico</h2>
      <ReplayPanel :runs="runs" :disabled="agentStatus === 'RUNNING'" @done="refresh(); notify('ok', 'Replay terminado')" @error="(m) => notify('error', m)" />
    </section>

    <div class="two">
      <section class="panel">
        <h2>Límites de riesgo</h2>
        <RiskPanel :risk="s.risk" :editable="agentStatus !== 'RUNNING'" @saved="refresh(); notify('ok', 'Límites guardados')" @error="(m) => notify('error', m)" />
      </section>
      <section class="panel">
        <h2>Bitácora de eventos y errores</h2>
        <div class="scroll">
          <table>
            <tbody>
              <tr v-for="e in events" :key="e.id">
                <td class="small muted nowrap">{{ time(e.ts) }}</td>
                <td :class="['small', `lvl-${e.level}`]">{{ e.kind }}</td>
                <td class="small">{{ e.message }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  </div>
  <div v-else class="page muted">Cargando…</div>
</template>

<style scoped>
.page { max-width: 1240px; margin: 0 auto; padding: 20px 16px 48px; display: grid; gap: 16px; }
header { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
h1 { font-size: 20px; }
.head-right { display: flex; gap: 8px; align-items: center; }
.paper { background: color-mix(in srgb, var(--warn) 18%, transparent); color: var(--warn); font-weight: 600; }
.st-RUNNING { background: color-mix(in srgb, var(--ok) 18%, transparent); color: var(--ok); }
.st-PAUSED, .st-HALTED { background: color-mix(in srgb, var(--warn) 18%, transparent); color: var(--warn); }
.st-STOPPED { background: color-mix(in srgb, var(--error) 15%, transparent); color: var(--error); }
.banner { margin: 0; padding: 10px 14px; border-radius: 10px; }
.banner.ok { background: color-mix(in srgb, var(--ok) 14%, transparent); }
.banner.error { background: color-mix(in srgb, var(--error) 14%, transparent); }
.reason { margin: -8px 0 0; }
.controls { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.sep { width: 1px; height: 24px; background: var(--line); margin: 0 4px; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
.card { display: grid; gap: 4px; align-content: start; }
.label { font-size: 12px; color: var(--muted); }
.big { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
.nowrap { white-space: nowrap; }
section.panel { overflow-x: auto; }
</style>
