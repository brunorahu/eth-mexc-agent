<script setup lang="ts">
import { ref } from 'vue'
import { api, apiKey, ApiError } from '../api'

const emit = defineEmits<{ ok: [] }>()
const key = ref('')
const error = ref('')
const loading = ref(false)

async function enter() {
  error.value = ''
  loading.value = true
  apiKey.set(key.value.trim())
  try {
    await api.status()
    emit('ok')
  } catch (e) {
    apiKey.clear()
    error.value = e instanceof ApiError && e.status === 401 ? 'API key incorrecta.' : `No se pudo conectar con el backend: ${(e as Error).message}`
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="gate">
    <form class="panel" @submit.prevent="enter">
      <h1>MEXC Paper Agent</h1>
      <p class="muted">Escribe la API key del backend (está en <code>backend/.env</code>). No se guarda en el código ni se envía a terceros.</p>
      <input v-model="key" type="password" placeholder="API key" autocomplete="off" required />
      <button class="primary" :disabled="loading || !key">{{ loading ? 'Verificando…' : 'Entrar' }}</button>
      <p v-if="error" class="lvl-error">{{ error }}</p>
    </form>
  </div>
</template>

<style scoped>
.gate { min-height: 100vh; display: grid; place-items: center; padding: 16px; }
form { width: min(420px, 100%); display: grid; gap: 12px; }
h1 { font-size: 20px; }
code { font-size: 12px; }
</style>
