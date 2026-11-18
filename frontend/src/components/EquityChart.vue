<script setup lang="ts">
import { computed } from 'vue'
import type { EquityRow } from '../types'
import { usd } from '../format'

const props = defineProps<{ rows: EquityRow[]; initial: number }>()
const W = 640
const H = 180
const P = 8

const geo = computed(() => {
  const vals = props.rows.map((r) => r.equity)
  if (vals.length < 2) return null
  const all = [...vals, props.initial]
  const min = Math.min(...all)
  const max = Math.max(...all)
  const span = max - min || 1
  const x = (i: number) => P + (i / (vals.length - 1)) * (W - 2 * P)
  const y = (v: number) => H - P - ((v - min) / span) * (H - 2 * P)
  const line = vals.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
  return { line, base: y(props.initial), min, max, last: vals[vals.length - 1] }
})
</script>

<template>
  <div>
    <div v-if="!geo" class="muted small">Aún no hay suficientes ciclos para graficar el patrimonio.</div>
    <template v-else>
      <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="none" role="img" aria-label="Patrimonio por ciclo">
        <line :x1="P" :x2="W - P" :y1="geo.base" :y2="geo.base" class="base" />
        <path :d="geo.line" :class="geo.last >= initial ? 'up' : 'down'" />
      </svg>
      <div class="legend small muted">
        <span>{{ rows[0].candle_time }}</span>
        <span>mín {{ usd(geo.min, 0) }} · máx {{ usd(geo.max, 0) }} · línea punteada = capital inicial</span>
        <span>{{ rows[rows.length - 1].candle_time }}</span>
      </div>
    </template>
  </div>
</template>

<style scoped>
svg { width: 100%; height: 180px; display: block; }
path { fill: none; stroke-width: 2; vector-effect: non-scaling-stroke; }
path.up { stroke: var(--buy); }
path.down { stroke: var(--sell); }
.base { stroke: var(--muted); stroke-dasharray: 4 4; vector-effect: non-scaling-stroke; }
.legend { display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap; margin-top: 4px; }
</style>
