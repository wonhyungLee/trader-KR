import axios from 'axios';

// Default: call API on the same origin root (works even if the frontend is served under /bnf).
// If your deployment proxies API under a prefix, set VITE_API_BASE (e.g. "/bnf").
const baseURL = import.meta.env.VITE_API_BASE ? import.meta.env.VITE_API_BASE : '';
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
