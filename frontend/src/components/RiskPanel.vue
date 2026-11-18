<script setup lang="ts">
import { reactive, watch, ref } from 'vue'
import { api } from '../api'
import type { RiskConfig } from '../types'

const props = defineProps<{ risk: RiskConfig; editable: boolean }>()
const emit = defineEmits<{ saved: []; error: [string] }>()

const form = reactive({ ...props.risk })
const saving = ref(false)
watch(() => props.risk, (r) => Object.assign(form, r))

const fields: { key: keyof RiskConfig; label: string; pct?: boolean }[] = [
  { key: 'max_exposure_pct', label: 'Exposición máxima', pct: true },
  { key: 'max_daily_loss_pct', label: 'Pérdida diaria máxima', pct: true },
  { key: 'max_data_age_hours', label: 'Antigüedad máx. del dato (h)' },
  { key: 'fee_rate', label: 'Comisión por operación', pct: true },
  { key: 'slippage_bps', label: 'Deslizamiento (pb)' },
  { key: 'signal_threshold', label: 'Umbral δ del modelo', pct: true },
  { key: 'min_notional', label: 'Orden mínima (USDT)' },
  { key: 'initial_capital', label: 'Capital inicial (USDT)' },
]

async function save() {
  saving.value = true
  try {
    await api.updateRisk({ ...form })
    emit('saved')
  } catch (e) {
    emit('error', (e as Error).message)
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <form @submit.prevent="save">
    <p class="muted small">
      Límites deterministas. El LLM no puede modificarlos; solo se cambian aquí con la API key y con el agente en pausa o detenido.
    </p>
    <div class="grid">
      <label v-for="f in fields" :key="f.key">
        <span>{{ f.label }}<span v-if="f.pct" class="muted"> (fracción)</span></span>
        <input v-model.number="form[f.key]" type="number" step="any" :disabled="!editable" />
      </label>
    </div>
    <button class="primary" :disabled="!editable || saving">{{ saving ? 'Guardando…' : 'Guardar límites' }}</button>
    <span v-if="!editable" class="small muted"> Pausa o detén el agente para editar.</span>
  </form>
</template>

<style scoped>
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; margin-bottom: 12px; }
label { display: grid; gap: 4px; font-size: 12px; color: var(--muted); }
input { width: 100%; }
</style>
