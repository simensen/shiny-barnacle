# Toolbridge Session Viewer - Complete Project Plan

A standalone Vite + TypeScript SPA for viewing and debugging toolbridge sessions.

## Quick Start

```bash
# Phase 1: Create and setup project
npm create vite@latest toolbridge-viewer -- --template vanilla-ts
cd toolbridge-viewer
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# Phase 2: Start development
npm run dev
```

---

## Phase 1: Project Setup

### 1.1 Create Project

```bash
npm create vite@latest toolbridge-viewer -- --template vanilla-ts
cd toolbridge-viewer
npm install
```

### 1.2 Install Dependencies

```bash
# Tailwind CSS for styling
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

### 1.3 Configure Tailwind

**tailwind.config.js**
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
}
```

### 1.4 Configure Vite

**vite.config.ts**
```typescript
import { defineConfig } from 'vite'

export default defineConfig({
  base: './', // Allows serving from any subdirectory
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
  },
  server: {
    port: 5173,
    // Optional: proxy for development to avoid CORS
    // proxy: {
    //   '/api': {
    //     target: 'http://localhost:4000',
    //     changeOrigin: true,
    //     rewrite: (path) => path.replace(/^\/api/, '/admin'),
    //   },
    // },
  },
})
```

### 1.5 Configure TypeScript

**tsconfig.json**
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "module": "ESNext",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src"]
}
```

---

## Phase 2: Project Structure

Create the following directory structure:

```bash
mkdir -p src/{types,api,stores,components,utils}
touch src/types/toolbridge.ts
touch src/api/client.ts
touch src/stores/config.ts
touch src/components/{ApiConfig,SessionList,SessionDetail,MessageList,ToolCallView}.ts
touch src/utils/{time,json}.ts
```

Final structure:
```
toolbridge-viewer/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.js
├── postcss.config.js
├── index.html
├── public/
│   └── favicon.svg
└── src/
    ├── main.ts
    ├── style.css
    ├── types/
    │   └── toolbridge.ts
    ├── api/
    │   └── client.ts
    ├── stores/
    │   └── config.ts
    ├── components/
    │   ├── ApiConfig.ts
    │   ├── SessionList.ts
    │   ├── SessionDetail.ts
    │   ├── MessageList.ts
    │   └── ToolCallView.ts
    └── utils/
        ├── time.ts
        └── json.ts
```

---

## Phase 3: Core Implementation

### 3.1 Types (copy from toolbridge/schemas/admin-sessions.d.ts)

**src/types/toolbridge.ts**
```typescript
/**
 * Toolbridge Admin Sessions API Types
 */

export interface ToolCallFunction {
  name: string;
  arguments: string;
}

export interface ToolCall {
  id: string;
  type: "function";
  function: ToolCallFunction;
}

export interface ChatMessage {
  timestamp: number;
  direction: "request" | "response";
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  tool_calls?: ToolCall[] | null;
  raw_content?: string | null;
}

export interface SessionSummary {
  created_at: number;
  last_seen_at: number;
  age_seconds: number;
  idle_seconds: number;
  request_count: number;
  tool_calls_total: number;
  tool_calls_fixed: number;
  tool_calls_failed: number;
  fix_rate: number;
  client_ip: string | null;
}

export interface SessionListResponse {
  active_sessions: number;
  session_timeout_seconds: number;
  sessions: Record<string, SessionSummary>;
}

export interface SessionDetailResponse {
  session_id: string;
  created_at: number;
  last_seen_at: number;
  age_seconds: number;
  idle_seconds: number;
  request_count: number;
  tool_calls_total: number;
  tool_calls_fixed: number;
  tool_calls_failed: number;
  fix_rate: number;
  client_ip: string | null;
  message_count: number;
  messages?: ChatMessage[];
}

export interface ErrorResponse {
  error: string;
  session_id?: string;
}
```

### 3.2 Configuration Store

**src/stores/config.ts**
```typescript
const STORAGE_KEY = 'toolbridge-api-url'

export function getApiUrl(): string | null {
  return localStorage.getItem(STORAGE_KEY)
}

export function setApiUrl(url: string): void {
  // Normalize URL - remove trailing slash
  const normalized = url.replace(/\/+$/, '')
  localStorage.setItem(STORAGE_KEY, normalized)
}

export function clearApiUrl(): void {
  localStorage.removeItem(STORAGE_KEY)
}
```

### 3.3 API Client

**src/api/client.ts**
```typescript
import type { SessionListResponse, SessionDetailResponse } from '../types/toolbridge'

export class ToolbridgeClient {
  constructor(private baseUrl: string) {}

  private async fetch<T>(path: string): Promise<T> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      headers: {
        'Accept': 'application/json',
      },
    })

    if (!res.ok) {
      const error = await res.json().catch(() => ({ error: `HTTP ${res.status}` }))
      throw new Error(error.error || `HTTP ${res.status}`)
    }

    return res.json()
  }

  async getSessions(): Promise<SessionListResponse> {
    return this.fetch('/admin/sessions')
  }

  async getSession(
    id: string,
    includeMessages = true,
    messageLimit = 100
  ): Promise<SessionDetailResponse> {
    const params = new URLSearchParams({
      include_messages: String(includeMessages),
      message_limit: String(messageLimit),
    })
    return this.fetch(`/admin/sessions/${id}?${params}`)
  }

  async testConnection(): Promise<{ ok: boolean; error?: string }> {
    try {
      await this.getSessions()
      return { ok: true }
    } catch (e) {
      return { ok: false, error: e instanceof Error ? e.message : String(e) }
    }
  }
}
```

### 3.4 Utilities

**src/utils/time.ts**
```typescript
/**
 * Format seconds into human-readable duration
 */
export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.floor(seconds)}s`
  }
  if (seconds < 3600) {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}m ${secs}s`
  }
  const hours = Math.floor(seconds / 3600)
  const mins = Math.floor((seconds % 3600) / 60)
  return `${hours}h ${mins}m`
}

/**
 * Format Unix timestamp to locale string
 */
export function formatTime(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleString()
}

/**
 * Format Unix timestamp to relative time (e.g., "2 minutes ago")
 */
export function formatRelativeTime(timestamp: number): string {
  const seconds = Math.floor(Date.now() / 1000 - timestamp)

  if (seconds < 60) return 'just now'
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return `${Math.floor(seconds / 86400)}d ago`
}
```

**src/utils/json.ts**
```typescript
/**
 * Safely parse JSON with fallback
 */
export function safeJsonParse(str: string): unknown {
  try {
    return JSON.parse(str)
  } catch {
    return null
  }
}

/**
 * Pretty print JSON with syntax highlighting classes
 */
export function prettyJson(obj: unknown, indent = 2): string {
  return JSON.stringify(obj, null, indent)
}

/**
 * Truncate string with ellipsis
 */
export function truncate(str: string, maxLength: number): string {
  if (str.length <= maxLength) return str
  return str.slice(0, maxLength - 3) + '...'
}
```

---

## Phase 4: Components

### 4.1 API Configuration

**src/components/ApiConfig.ts**
```typescript
import { setApiUrl } from '../stores/config'
import { ToolbridgeClient } from '../api/client'

export function renderApiConfig(error?: string): string {
  return `
    <div class="min-h-screen flex items-center justify-center bg-gray-100 dark:bg-gray-900 p-4">
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 max-w-md w-full">
        <h1 class="text-2xl font-bold text-gray-900 dark:text-white mb-2">
          Toolbridge Session Viewer
        </h1>
        <p class="text-gray-600 dark:text-gray-400 mb-6">
          Connect to a toolbridge instance to view sessions.
        </p>

        ${error ? `
          <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-3 mb-4">
            <p class="text-red-700 dark:text-red-400 text-sm">${error}</p>
          </div>
        ` : ''}

        <form id="api-config-form" class="space-y-4">
          <div>
            <label for="api-url" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Toolbridge URL
            </label>
            <input
              type="url"
              id="api-url"
              name="api-url"
              placeholder="http://localhost:4000"
              required
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md
                     bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                     focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          <button
            type="submit"
            class="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md
                   transition-colors duration-200"
          >
            Connect
          </button>
        </form>

        <p class="mt-4 text-xs text-gray-500 dark:text-gray-400">
          Make sure toolbridge is running with <code class="bg-gray-100 dark:bg-gray-700 px-1 rounded">--cors</code> flag.
        </p>
      </div>
    </div>
  `
}

export function setupApiConfigHandlers(onConnect: (url: string) => void): void {
  const form = document.getElementById('api-config-form') as HTMLFormElement
  const input = document.getElementById('api-url') as HTMLInputElement

  form?.addEventListener('submit', async (e) => {
    e.preventDefault()
    const url = input.value.trim()

    if (!url) return

    // Test connection
    const client = new ToolbridgeClient(url)
    const result = await client.testConnection()

    if (result.ok) {
      setApiUrl(url)
      onConnect(url)
    } else {
      // Re-render with error
      const app = document.getElementById('app')!
      app.innerHTML = renderApiConfig(`Connection failed: ${result.error}`)
      setupApiConfigHandlers(onConnect)
    }
  })
}
```

### 4.2 Session List

**src/components/SessionList.ts**
```typescript
import type { SessionListResponse } from '../types/toolbridge'
import { formatDuration, formatRelativeTime } from '../utils/time'

export function renderSessionList(data: SessionListResponse, apiUrl: string): string {
  const sessions = Object.entries(data.sessions)
    .sort(([, a], [, b]) => b.last_seen_at - a.last_seen_at)

  const rows = sessions.length === 0
    ? `<tr><td colspan="7" class="px-4 py-8 text-center text-gray-500">No active sessions</td></tr>`
    : sessions.map(([id, s]) => `
        <tr class="border-b border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 cursor-pointer"
            onclick="location.hash='/session/${id}'">
          <td class="px-4 py-3 font-mono text-sm text-blue-600 dark:text-blue-400">${id}</td>
          <td class="px-4 py-3 text-gray-600 dark:text-gray-400">${s.client_ip || '-'}</td>
          <td class="px-4 py-3 text-gray-600 dark:text-gray-400">${formatDuration(s.age_seconds)}</td>
          <td class="px-4 py-3 text-gray-600 dark:text-gray-400">${formatRelativeTime(s.last_seen_at)}</td>
          <td class="px-4 py-3 text-gray-900 dark:text-white">${s.request_count}</td>
          <td class="px-4 py-3">
            <span class="text-green-600 dark:text-green-400">${s.tool_calls_fixed}</span>
            <span class="text-gray-400">/</span>
            <span class="text-gray-600 dark:text-gray-400">${s.tool_calls_total}</span>
          </td>
          <td class="px-4 py-3">
            ${renderFixRate(s.fix_rate, s.tool_calls_total)}
          </td>
        </tr>
      `).join('')

  return `
    <div class="min-h-screen bg-gray-100 dark:bg-gray-900">
      <header class="bg-white dark:bg-gray-800 shadow">
        <div class="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
          <div>
            <h1 class="text-xl font-bold text-gray-900 dark:text-white">Toolbridge Sessions</h1>
            <p class="text-sm text-gray-500 dark:text-gray-400">${apiUrl}</p>
          </div>
          <div class="flex items-center gap-4">
            <span class="text-sm text-gray-600 dark:text-gray-400">
              ${data.active_sessions} active · timeout ${data.session_timeout_seconds}s
            </span>
            <button id="refresh-btn"
                    class="px-3 py-1.5 text-sm bg-gray-200 dark:bg-gray-700 hover:bg-gray-300
                           dark:hover:bg-gray-600 rounded-md transition-colors">
              Refresh
            </button>
            <button id="disconnect-btn"
                    class="px-3 py-1.5 text-sm text-red-600 hover:text-red-700
                           dark:text-red-400 dark:hover:text-red-300">
              Disconnect
            </button>
          </div>
        </div>
      </header>

      <main class="max-w-7xl mx-auto px-4 py-6">
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
          <table class="w-full">
            <thead class="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Session ID</th>
                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Client IP</th>
                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Age</th>
                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Last Seen</th>
                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Requests</th>
                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Tool Calls</th>
                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Fix Rate</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-200 dark:divide-gray-700">
              ${rows}
            </tbody>
          </table>
        </div>
      </main>
    </div>
  `
}

function renderFixRate(rate: number, total: number): string {
  if (total === 0) {
    return '<span class="text-gray-400">-</span>'
  }

  const percentage = Math.round(rate * 100)
  const colorClass = percentage === 0
    ? 'text-gray-600 dark:text-gray-400'
    : percentage < 50
      ? 'text-yellow-600 dark:text-yellow-400'
      : 'text-green-600 dark:text-green-400'

  return `<span class="${colorClass}">${percentage}%</span>`
}

export function setupSessionListHandlers(
  onRefresh: () => void,
  onDisconnect: () => void
): void {
  document.getElementById('refresh-btn')?.addEventListener('click', onRefresh)
  document.getElementById('disconnect-btn')?.addEventListener('click', onDisconnect)
}
```

### 4.3 Session Detail

**src/components/SessionDetail.ts**
```typescript
import type { SessionDetailResponse } from '../types/toolbridge'
import { formatDuration, formatTime } from '../utils/time'
import { renderMessageList } from './MessageList'

export function renderSessionDetail(session: SessionDetailResponse): string {
  return `
    <div class="min-h-screen bg-gray-100 dark:bg-gray-900">
      <header class="bg-white dark:bg-gray-800 shadow">
        <div class="max-w-7xl mx-auto px-4 py-4">
          <div class="flex items-center gap-4">
            <a href="#/" class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
              ← Back
            </a>
            <h1 class="text-xl font-bold text-gray-900 dark:text-white font-mono">
              ${session.session_id}
            </h1>
          </div>
        </div>
      </header>

      <main class="max-w-7xl mx-auto px-4 py-6 space-y-6">
        <!-- Stats Cards -->
        <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          ${renderStatCard('Client IP', session.client_ip || '-')}
          ${renderStatCard('Created', formatTime(session.created_at))}
          ${renderStatCard('Age', formatDuration(session.age_seconds))}
          ${renderStatCard('Idle', formatDuration(session.idle_seconds))}
          ${renderStatCard('Requests', String(session.request_count))}
          ${renderStatCard('Messages', String(session.message_count))}
        </div>

        <!-- Tool Call Stats -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Tool Call Statistics</h2>
          <div class="grid grid-cols-4 gap-4">
            ${renderToolStat('Total', session.tool_calls_total, 'text-gray-600 dark:text-gray-400')}
            ${renderToolStat('Fixed', session.tool_calls_fixed, 'text-green-600 dark:text-green-400')}
            ${renderToolStat('Failed', session.tool_calls_failed, 'text-red-600 dark:text-red-400')}
            ${renderToolStat('Fix Rate', session.tool_calls_total > 0
              ? `${Math.round(session.fix_rate * 100)}%`
              : '-', 'text-blue-600 dark:text-blue-400')}
          </div>
        </div>

        <!-- Message History -->
        ${session.messages ? renderMessageList(session.messages) : `
          <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6 text-center text-gray-500">
            No message history available
          </div>
        `}
      </main>
    </div>
  `
}

function renderStatCard(label: string, value: string): string {
  return `
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
      <dt class="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">${label}</dt>
      <dd class="mt-1 text-lg font-semibold text-gray-900 dark:text-white truncate">${value}</dd>
    </div>
  `
}

function renderToolStat(label: string, value: number | string, colorClass: string): string {
  return `
    <div class="text-center">
      <div class="text-2xl font-bold ${colorClass}">${value}</div>
      <div class="text-sm text-gray-500 dark:text-gray-400">${label}</div>
    </div>
  `
}
```

### 4.4 Message List

**src/components/MessageList.ts**
```typescript
import type { ChatMessage } from '../types/toolbridge'
import { formatTime } from '../utils/time'
import { renderToolCallView } from './ToolCallView'

export function renderMessageList(messages: ChatMessage[]): string {
  if (messages.length === 0) {
    return `
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6 text-center text-gray-500">
        No messages recorded
      </div>
    `
  }

  return `
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow">
      <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <h2 class="text-lg font-semibold text-gray-900 dark:text-white">
          Message History
          <span class="text-sm font-normal text-gray-500 dark:text-gray-400 ml-2">
            (${messages.length} messages)
          </span>
        </h2>
      </div>
      <div class="divide-y divide-gray-200 dark:divide-gray-700">
        ${messages.map(renderMessage).join('')}
      </div>
    </div>
  `
}

function renderMessage(msg: ChatMessage): string {
  const directionClass = msg.direction === 'request'
    ? 'border-l-blue-500'
    : 'border-l-green-500'

  const roleColors: Record<string, string> = {
    user: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    assistant: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    system: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
    tool: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
  }

  const hasTransformation = msg.raw_content && msg.raw_content !== msg.content

  return `
    <div class="p-4 border-l-4 ${directionClass}">
      <div class="flex items-center gap-2 mb-2">
        <span class="px-2 py-0.5 text-xs font-medium rounded ${roleColors[msg.role] || roleColors.user}">
          ${msg.role}
        </span>
        <span class="text-xs text-gray-500 dark:text-gray-400">
          ${formatTime(msg.timestamp)}
        </span>
        <span class="text-xs text-gray-400 dark:text-gray-500">
          ${msg.direction}
        </span>
        ${hasTransformation ? `
          <span class="px-2 py-0.5 text-xs font-medium rounded bg-yellow-100 text-yellow-800
                       dark:bg-yellow-900 dark:text-yellow-200">
            transformed
          </span>
        ` : ''}
      </div>

      <div class="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap font-mono bg-gray-50
                  dark:bg-gray-900 rounded p-3 max-h-96 overflow-auto">
        ${escapeHtml(msg.content)}
      </div>

      ${hasTransformation ? `
        <details class="mt-2">
          <summary class="text-xs text-gray-500 dark:text-gray-400 cursor-pointer hover:text-gray-700
                         dark:hover:text-gray-300">
            Show original (before transformation)
          </summary>
          <div class="mt-2 text-sm text-gray-600 dark:text-gray-400 whitespace-pre-wrap font-mono
                      bg-red-50 dark:bg-red-900/20 rounded p-3 max-h-96 overflow-auto border
                      border-red-200 dark:border-red-800">
            ${escapeHtml(msg.raw_content || '')}
          </div>
        </details>
      ` : ''}

      ${msg.tool_calls && msg.tool_calls.length > 0 ? renderToolCallView(msg.tool_calls) : ''}
    </div>
  `
}

function escapeHtml(str: string): string {
  const div = document.createElement('div')
  div.textContent = str
  return div.innerHTML
}
```

### 4.5 Tool Call View

**src/components/ToolCallView.ts**
```typescript
import type { ToolCall } from '../types/toolbridge'
import { safeJsonParse, prettyJson } from '../utils/json'

export function renderToolCallView(toolCalls: ToolCall[]): string {
  return `
    <div class="mt-3 space-y-2">
      <div class="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
        Tool Calls (${toolCalls.length})
      </div>
      ${toolCalls.map(renderToolCall).join('')}
    </div>
  `
}

function renderToolCall(tc: ToolCall): string {
  const parsedArgs = safeJsonParse(tc.function.arguments)
  const formattedArgs = parsedArgs
    ? prettyJson(parsedArgs)
    : tc.function.arguments

  return `
    <div class="bg-gray-100 dark:bg-gray-700 rounded-lg p-3">
      <div class="flex items-center gap-2 mb-2">
        <span class="font-mono text-sm font-semibold text-purple-600 dark:text-purple-400">
          ${escapeHtml(tc.function.name)}
        </span>
        <span class="text-xs text-gray-400 dark:text-gray-500 font-mono">
          ${tc.id}
        </span>
      </div>
      <pre class="text-xs text-gray-600 dark:text-gray-300 overflow-x-auto"><code>${escapeHtml(formattedArgs)}</code></pre>
    </div>
  `
}

function escapeHtml(str: string): string {
  const div = document.createElement('div')
  div.textContent = str
  return div.innerHTML
}
```

---

## Phase 5: Main Application

### 5.1 Entry HTML

**index.html**
```html
<!DOCTYPE html>
<html lang="en" class="">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Toolbridge Session Viewer</title>
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
  </head>
  <body class="bg-gray-100 dark:bg-gray-900">
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

### 5.2 Styles

**src/style.css**
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

/* Custom scrollbar for dark mode */
@layer utilities {
  .dark ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }

  .dark ::-webkit-scrollbar-track {
    background: #1f2937;
  }

  .dark ::-webkit-scrollbar-thumb {
    background: #4b5563;
    border-radius: 4px;
  }

  .dark ::-webkit-scrollbar-thumb:hover {
    background: #6b7280;
  }
}
```

### 5.3 Main Entry Point

**src/main.ts**
```typescript
import './style.css'
import { getApiUrl, clearApiUrl } from './stores/config'
import { ToolbridgeClient } from './api/client'
import { renderApiConfig, setupApiConfigHandlers } from './components/ApiConfig'
import { renderSessionList, setupSessionListHandlers } from './components/SessionList'
import { renderSessionDetail } from './components/SessionDetail'

// Check for dark mode preference
if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
  document.documentElement.classList.add('dark')
}

// Simple hash-based router
function getRoute(): { view: 'config' | 'list' | 'session'; id?: string } {
  const apiUrl = getApiUrl()
  if (!apiUrl) {
    return { view: 'config' }
  }

  const hash = window.location.hash.slice(1) || '/'

  if (hash.startsWith('/session/')) {
    const id = hash.slice(9)
    return { view: 'session', id }
  }

  return { view: 'list' }
}

// Render the app
async function render() {
  const app = document.getElementById('app')!
  const route = getRoute()

  // Config screen
  if (route.view === 'config') {
    app.innerHTML = renderApiConfig()
    setupApiConfigHandlers(() => render())
    return
  }

  const apiUrl = getApiUrl()!
  const client = new ToolbridgeClient(apiUrl)

  try {
    if (route.view === 'session' && route.id) {
      // Session detail view
      app.innerHTML = '<div class="p-8 text-center text-gray-500">Loading session...</div>'
      const session = await client.getSession(route.id, true, 500)
      app.innerHTML = renderSessionDetail(session)
    } else {
      // Session list view
      app.innerHTML = '<div class="p-8 text-center text-gray-500">Loading sessions...</div>'
      const data = await client.getSessions()
      app.innerHTML = renderSessionList(data, apiUrl)
      setupSessionListHandlers(
        () => render(),
        () => {
          clearApiUrl()
          render()
        }
      )
    }
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err)
    app.innerHTML = `
      <div class="min-h-screen flex items-center justify-center bg-gray-100 dark:bg-gray-900">
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 max-w-md">
          <h2 class="text-xl font-bold text-red-600 dark:text-red-400 mb-2">Connection Error</h2>
          <p class="text-gray-600 dark:text-gray-400 mb-4">${error}</p>
          <div class="flex gap-2">
            <button onclick="location.reload()"
                    class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
              Retry
            </button>
            <button id="reconfigure-btn"
                    class="px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded hover:bg-gray-300
                           dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300">
              Reconfigure
            </button>
          </div>
        </div>
      </div>
    `
    document.getElementById('reconfigure-btn')?.addEventListener('click', () => {
      clearApiUrl()
      render()
    })
  }
}

// Listen for hash changes
window.addEventListener('hashchange', render)

// Initial render
render()
```

---

## Phase 6: Optional Enhancements

### 6.1 Auto-refresh

Add to `src/main.ts` or create `src/utils/autorefresh.ts`:

```typescript
let refreshInterval: number | null = null

export function startAutoRefresh(callback: () => void, intervalMs = 5000): void {
  stopAutoRefresh()
  refreshInterval = window.setInterval(callback, intervalMs)
}

export function stopAutoRefresh(): void {
  if (refreshInterval !== null) {
    window.clearInterval(refreshInterval)
    refreshInterval = null
  }
}
```

### 6.2 Favicon

**public/favicon.svg**
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect width="100" height="100" rx="20" fill="#3b82f6"/>
  <path d="M30 35h40M30 50h40M30 65h25" stroke="white" stroke-width="8" stroke-linecap="round"/>
</svg>
```

---

## Phase 7: Build & Deploy

### 7.1 Development

```bash
npm run dev
# Opens at http://localhost:5173
```

### 7.2 Build for Production

```bash
npm run build
npm run preview  # Test production build locally
```

### 7.3 Deployment Options

**Option A: Static hosting (GitHub Pages, Netlify, Vercel)**
```bash
# Build outputs to dist/
npm run build

# Deploy dist/ folder to your hosting provider
```

**Option B: Serve from any HTTP server**
```bash
npm run build
npx serve dist
# Or: python -m http.server -d dist
```

**Option C: Docker**
```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
```

---

## Commands Summary

```bash
# === SETUP ===
npm create vite@latest toolbridge-viewer -- --template vanilla-ts
cd toolbridge-viewer
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# === CREATE STRUCTURE ===
mkdir -p src/{types,api,stores,components,utils}

# === DEVELOPMENT ===
npm run dev              # Start dev server (http://localhost:5173)

# === BUILD ===
npm run build            # Build to dist/
npm run preview          # Preview production build

# === TYPE CHECK ===
npx tsc --noEmit         # Check types without building

# === DEPLOY ===
npx serve dist           # Quick local static server
```

---

## Toolbridge Requirements

For the viewer to connect, start toolbridge with CORS enabled:

```bash
# Allow any origin (development)
python toolbridge.py --cors

# Specific origin (production)
python toolbridge.py --cors --cors-origins "https://your-viewer-domain.com"
```
