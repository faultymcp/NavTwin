const API_BASE = process.env.EXPO_PUBLIC_API_BASE || 'http://192.168.0.39:8000';

async function api(path, { method = "GET", token, body } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(`${API_BASE}${path}`, {  // ✅ FIXED: Changed BASE to API_BASE
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

export { api, API_BASE };  // ✅ FIXED: Export API_BASE, not BASE