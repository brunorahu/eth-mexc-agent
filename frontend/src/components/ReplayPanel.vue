<script setup lang="ts">
import { computed, ref } from 'vue'
import { api } from '../api'
import type { Run } from '../types'
import { pct, time, usd } from '../format'

const props = defineProps<{ runs: Run[]; disabled: boolean }>()
const emit = defineEmits<{ done: []; error: [string] }>()

const start = ref('2026-10-01')
const end = ref('2026-10-31')
const limit = ref(20)
const running = ref(false)

const strategyName: Record<string, string> = {
  agent_llm: 'Agente con LLM',
  model_only: 'Modelo sin LLM',
  buy_hold: 'Comprar y mantener',
}

const replays = computed(() => {
  const groups = new Map<string, Run[]>()
  for (const r of props.runs.filter((r) => r.mode === 'replay')) {
    const tag = r.id.split('-').slice(1, 2)[0]
    groups.set(tag, [...(groups.get(tag) ?? []), r])
  }
  return [...groups.entries()].slice(0, 5)
})

async function run() {
  running.value = true
  try {
    await api.replay({ start: start.value, end: end.value, limit: limit.value, strategies: ['agent_llm', 'model_only', 'buy_hold'] })
    emit('done')
  } catch (e) {
    emit('error', (e as Error).message)
  } finally {
    running.value = false
  }
}
</script>

<template>
  <div>
    <p class="muted small">
      Repite decisiones con velas históricas de MEXC. Está marcado como <b>REPLAY</b> y separado del modo en vivo:
      decisión al cierre de t, ejecución a la apertura de t+1. Las tres estrategias usan el mismo periodo y los mismos costos.
    </p>
    <form class="row" @submit.prevent="run">
      <label>Desde <input v-model="start" type="date" required /></label>
      <label>Hasta <input v-model="end" type="date" required /></label>
      <label>Velas <input v-model.number="limit" type="number" min="1" max="1000" required /></label>
      <button class="primary" :disabled="running || disabled">{{ running ? 'Ejecutando… (≈4 s por vela)' : 'Ejecutar replay' }}</button>
    </form>
    <p v-if="disabled" class="small lvl-warning">Pausa o detén el agente en vivo para ejecutar un replay.</p>

    <div v-for="[tag, rs] in replays" :key="tag" class="group">
      <div class="small muted">REPLAY {{ tag }} · {{ time(rs[0].created_at) }} · {{ rs[0].metrics.desde }} → {{ rs[0].metrics.hasta }}</div>
      <table>
        <thead>
          <tr>
            <th>Estrategia</th>
            <th class="num">Ciclos</th>
            <th class="num">Rend. neto</th>
            <th class="num">Máx. DD</th>
            <th class="num">Compras/Ventas</th>
            <th class="num">Win Rate</th>
            <th class="num">Costos</th>
            <th class="num">Rechazadas</th>
            <th class="num">Latencia LLM</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="r in rs" :key="r.id">
            <td>{{ strategyName[r.strategy] }}</td>
            <td class="num">{{ r.metrics.ciclos }}</td>
            <td class="num">{{ pct(r.metrics.rendimiento_neto) }}</td>
            <td class="num">{{ pct(r.metrics.max_drawdown) }}</td>
            <td class="num">{{ r.metrics.compras }}/{{ r.metrics.ventas }}</td>
            <td class="num">{{ r.metrics.win_rate == null ? 'no definido' : pct(r.metrics.win_rate, 0) }}</td>
            <td class="num">{{ usd(r.metrics.costos_totales) }}</td>
            <td class="num">{{ r.metrics.acciones_rechazadas }}</td>
            <td class="num">{{ r.metrics.latencia_llm_ms_media ? `${Math.round(r.metrics.latencia_llm_ms_media)} ms` : '—' }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.row { display: flex; gap: 10px; flex-wrap: wrap; align-items: end; margin-bottom: 12px; }
label { display: grid; gap: 4px; font-size: 12px; color: var(--muted); }
.group { margin-top: 14px; overflow-x: auto; }
</style>
