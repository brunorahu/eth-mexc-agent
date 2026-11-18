<script setup lang="ts">
import type { Decision } from '../types'
import { actionLabel, pct, usd } from '../format'

defineProps<{ rows: Decision[] }>()
</script>

<template>
  <div class="scroll">
    <p v-if="!rows.length" class="muted small">Sin decisiones todavía.</p>
    <table v-else>
      <thead>
        <tr>
          <th>Vela</th>
          <th class="num">Cierre</th>
          <th class="num">Pronóstico</th>
          <th>Propuesta LLM</th>
          <th>Final</th>
          <th>Motivo / rechazo</th>
          <th class="num">LLM ms</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="d in rows" :key="d.id">
          <td>{{ d.candle_time }}</td>
          <td class="num">{{ usd(d.close) }}</td>
          <td class="num">{{ pct(d.forecast, 3) }}</td>
          <td :class="`act-${d.proposed_action}`">{{ actionLabel[d.proposed_action] ?? d.proposed_action }}</td>
          <td :class="`act-${d.final_action}`">{{ actionLabel[d.final_action] ?? d.final_action }}</td>
          <td class="small">
            <div>{{ d.justification ?? '—' }}</div>
            <div v-if="d.rejected" class="lvl-warning">Rechazada: {{ d.reject_reason }}</div>
          </td>
          <td class="num">{{ d.llm_latency_ms ? Math.round(d.llm_latency_ms) : '—' }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
