import axios from 'axios';

// Vite 프록시 또는 Nginx 설정에 맞춰 /api 접두사 사용 여부 결정
// /bnf 또는 루트 둘 다 대응하도록 현재 경로 기반으로 기본값 결정
const inferredBase = window.location.pathname.startsWith('/bnf') ? '/bnf' : '';
const baseURL = import.meta.env.VITE_API_BASE ? import.meta.env.VITE_API_BASE : inferredBase;
const api = axios.create({ baseURL });

export const fetchUniverse = (sector) => api.get('/universe', { params: sector ? { sector } : {} }).then(r => r.data);
export const fetchSectors = () => api.get('/sectors').then(r => r.data);
export const fetchPrices = (code, days = 60) => api.get('/prices', { params: { code, days } }).then(r => r.data);
export const fetchRealtimePrice = (code) => api.get('/prices/realtime', { params: { code } }).then(r => r.data);
export const fetchSelectionRealtimePrices = (codes) =>
  api.get('/selection/realtime_prices', { params: { codes: Array.isArray(codes) ? codes.join(',') : '' } }).then(r => r.data);
export const fetchStatus = () => api.get('/status').then(r => r.data);
export const fetchSelection = () => api.get('/selection').then(r => r.data);
export const fetchSelectionFilters = () => api.get('/selection_filters').then(r => r.data);
export const updateSelectionFilterToggle = (key, enabled, password) =>
  api.post('/selection_filters/toggle', { key, enabled, password }).then(r => r.data);
export const fetchPortfolio = () => api.get('/portfolio').then(r => r.data);
export const fetchPlans = () => api.get('/plans').then(r => r.data);
export const fetchAccount = () => api.get('/account').then(r => r.data);
export const fetchKisKeys = () => api.get('/kis_keys').then(r => r.data);
export const updateKisKeyToggle = (id, enabled, password) =>
  api.post('/kis_keys/toggle', { id, enabled, password }).then(r => r.data);
