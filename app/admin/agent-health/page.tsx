"use client";

import { useEffect, useState, useCallback } from "react";

// ── Types ──────────────────────────────────────────────────────────────────────

interface AgentInfo {
  user_id: string;
  hostname: string;
  machine_id: string;
  agent_version: string;
  gpu_count: number;
  gpu_uuids: string[];
  gpu_names: string[];
  scheduler: string;
  uptime_sec: number;
  config_hash: string;
  cluster_tag: string;
  location_hint: string;
  last_seen: string;
  status: "online" | "offline";
  error_count_total: number;
  error_count_last_hour: number;
  last_error_message: string | null;
  last_error_at: string | null;
  os_info: string | null;
  python_version: string | null;
  gpu_backend: string | null;
  agent_mode: string | null;
}

interface AgentSummary {
  total: number;
  online: number;
  offline: number;
  versions: Array<{ version: string; count: number }>;
}

interface ErrorEntry {
  id: string;
  user_id: string;
  machine_id: string;
  hostname: string;
  error_type: string;
  error_message: string;
  stack_trace: string | null;
  gpu_index: number | null;
  created_at: string;
}

interface HealthAlert {
  id: string;
  user_id: string;
  machine_id: string;
  hostname: string;
  alert_type: string;
  severity: string;
  message: string;
  resolved: boolean;
  resolved_at: string | null;
  created_at: string;
}

type Tab = "fleet" | "errors" | "alerts" | "diagnostics";

// ── Helpers ────────────────────────────────────────────────────────────────────

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  if (diff < 60_000) return `${Math.floor(diff / 1000)}s ago`;
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return `${Math.floor(diff / 86_400_000)}d ago`;
}

function uptimeStr(sec: number): string {
  if (sec < 60) return `${Math.floor(sec)}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
  return `${Math.floor(sec / 86400)}d ${Math.floor((sec % 86400) / 3600)}h`;
}

const SEVERITY_COLORS: Record<string, string> = {
  critical: "bg-red-600/20 text-red-400 border-red-600/40",
  warning: "bg-yellow-600/20 text-yellow-400 border-yellow-600/40",
};

const ALERT_TYPE_LABELS: Record<string, string> = {
  agent_offline: "Agent Offline",
  version_mismatch: "Version Mismatch",
  config_drift: "Config Drift",
  high_error_rate: "High Error Rate",
};

const ERROR_TYPE_COLORS: Record<string, string> = {
  collection: "text-orange-400",
  upload: "text-blue-400",
  scheduler: "text-purple-400",
  power_control: "text-red-400",
  attribution: "text-cyan-400",
  auto_tuner: "text-green-400",
  unhandled: "text-red-500",
};

// ── Component ──────────────────────────────────────────────────────────────────

export default function AgentHealthPage() {
  const [tab, setTab] = useState<Tab>("fleet");
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [summary, setSummary] = useState<AgentSummary>({ total: 0, online: 0, offline: 0, versions: [] });
  const [errors, setErrors] = useState<ErrorEntry[]>([]);
  const [errorsTotal, setErrorsTotal] = useState(0);
  const [alerts, setAlerts] = useState<HealthAlert[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<AgentInfo | null>(null);
  const [agentErrors, setAgentErrors] = useState<ErrorEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [errorTypeFilter, setErrorTypeFilter] = useState<string>("");
  const [expandedError, setExpandedError] = useState<string | null>(null);
  const [sortField, setSortField] = useState<string>("last_seen");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  // ── Data fetching ────────────────────────────────────────────────────

  const fetchAgents = useCallback(async () => {
    const params = new URLSearchParams();
    if (statusFilter) params.set("status", statusFilter);
    const res = await fetch(`/api/admin/agent-health?${params}`);
    if (res.ok) {
      const data = await res.json();
      setAgents(data.agents ?? []);
      setSummary(data.summary ?? { total: 0, online: 0, offline: 0, versions: [] });
    }
  }, [statusFilter]);

  const fetchErrors = useCallback(async (machineId?: string) => {
    const params = new URLSearchParams({ limit: "200" });
    if (machineId) params.set("machine_id", machineId);
    if (errorTypeFilter) params.set("error_type", errorTypeFilter);
    const res = await fetch(`/api/admin/agent-health/errors?${params}`);
    if (res.ok) {
      const data = await res.json();
      if (machineId) {
        setAgentErrors(data.errors ?? []);
      } else {
        setErrors(data.errors ?? []);
        setErrorsTotal(data.total ?? 0);
      }
    }
  }, [errorTypeFilter]);

  const fetchAlerts = useCallback(async () => {
    const res = await fetch("/api/admin/agent-health/alerts");
    if (res.ok) {
      const data = await res.json();
      setAlerts(data.alerts ?? []);
    } else {
      // alerts endpoint might not exist yet — use agent_health_alerts via errors endpoint
      setAlerts([]);
    }
  }, []);

  useEffect(() => {
    setLoading(true);
    Promise.all([fetchAgents(), fetchErrors(), fetchAlerts()]).finally(() =>
      setLoading(false)
    );
    const interval = setInterval(() => {
      fetchAgents();
      fetchErrors();
      fetchAlerts();
    }, 30_000);
    return () => clearInterval(interval);
  }, [fetchAgents, fetchErrors, fetchAlerts]);

  // ── Sorting ──────────────────────────────────────────────────────────

  const toggleSort = (field: string) => {
    if (sortField === field) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDir("desc");
    }
  };

  const sortedAgents = [...agents].sort((a, b) => {
    const aVal = (a as unknown as Record<string, unknown>)[sortField];
    const bVal = (b as unknown as Record<string, unknown>)[sortField];
    if (aVal == null && bVal == null) return 0;
    if (aVal == null) return 1;
    if (bVal == null) return -1;
    const cmp =
      typeof aVal === "number"
        ? (aVal as number) - (bVal as number)
        : String(aVal).localeCompare(String(bVal));
    return sortDir === "asc" ? cmp : -cmp;
  });

  // ── Open diagnostics for a specific agent ────────────────────────────

  const openDiagnostics = (agent: AgentInfo) => {
    setSelectedAgent(agent);
    setTab("diagnostics");
    fetchErrors(agent.machine_id ?? agent.hostname);
  };

  // ── Tab: Fleet Overview ──────────────────────────────────────────────

  const renderFleet = () => (
    <div className="space-y-6">
      {/* Summary cards */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4">
          <div className="text-sm text-neutral-400">Total Agents</div>
          <div className="text-3xl font-bold text-white mt-1">{summary.total}</div>
        </div>
        <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4">
          <div className="text-sm text-neutral-400">Online</div>
          <div className="text-3xl font-bold text-green-400 mt-1">{summary.online}</div>
        </div>
        <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4">
          <div className="text-sm text-neutral-400">Offline</div>
          <div className="text-3xl font-bold text-red-400 mt-1">{summary.offline}</div>
        </div>
        <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4">
          <div className="text-sm text-neutral-400">Versions</div>
          <div className="mt-1 space-y-1">
            {summary.versions.map((v) => (
              <div key={v.version} className="flex justify-between text-sm">
                <span className="text-neutral-300">v{v.version}</span>
                <span className="text-neutral-500">{v.count}</span>
              </div>
            ))}
            {summary.versions.length === 0 && (
              <div className="text-sm text-neutral-500">No agents</div>
            )}
          </div>
        </div>
      </div>

      {/* Filter */}
      <div className="flex gap-2">
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-white"
        >
          <option value="">All Status</option>
          <option value="online">Online</option>
          <option value="offline">Offline</option>
        </select>
      </div>

      {/* Agent table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-neutral-800 text-neutral-400">
              <th className="text-left py-3 px-3">Status</th>
              <th
                className="text-left py-3 px-3 cursor-pointer hover:text-white"
                onClick={() => toggleSort("hostname")}
              >
                Hostname {sortField === "hostname" && (sortDir === "asc" ? "↑" : "↓")}
              </th>
              <th className="text-left py-3 px-3">GPUs</th>
              <th className="text-left py-3 px-3">Backend</th>
              <th
                className="text-left py-3 px-3 cursor-pointer hover:text-white"
                onClick={() => toggleSort("agent_version")}
              >
                Version {sortField === "agent_version" && (sortDir === "asc" ? "↑" : "↓")}
              </th>
              <th
                className="text-left py-3 px-3 cursor-pointer hover:text-white"
                onClick={() => toggleSort("uptime_sec")}
              >
                Uptime {sortField === "uptime_sec" && (sortDir === "asc" ? "↑" : "↓")}
              </th>
              <th className="text-left py-3 px-3">Errors/hr</th>
              <th
                className="text-left py-3 px-3 cursor-pointer hover:text-white"
                onClick={() => toggleSort("last_seen")}
              >
                Last Seen {sortField === "last_seen" && (sortDir === "asc" ? "↑" : "↓")}
              </th>
              <th className="text-left py-3 px-3">Config</th>
              <th className="text-left py-3 px-3"></th>
            </tr>
          </thead>
          <tbody>
            {sortedAgents.map((agent) => (
              <tr
                key={`${agent.user_id}-${agent.hostname}`}
                className="border-b border-neutral-800/50 hover:bg-neutral-900/50"
              >
                <td className="py-3 px-3">
                  <span
                    className={`inline-block w-2.5 h-2.5 rounded-full ${
                      agent.status === "online" ? "bg-green-500" : "bg-red-500"
                    }`}
                  />
                </td>
                <td className="py-3 px-3 text-white font-mono">{agent.hostname}</td>
                <td className="py-3 px-3 text-neutral-300">
                  {agent.gpu_count}x{" "}
                  {agent.gpu_names?.length > 0
                    ? agent.gpu_names[0]
                    : "unknown"}
                </td>
                <td className="py-3 px-3 text-neutral-400">{agent.gpu_backend ?? "-"}</td>
                <td className="py-3 px-3">
                  <span className="text-neutral-300">v{agent.agent_version}</span>
                </td>
                <td className="py-3 px-3 text-neutral-400">{uptimeStr(agent.uptime_sec)}</td>
                <td className="py-3 px-3">
                  <span
                    className={
                      (agent.error_count_last_hour ?? 0) > 10
                        ? "text-red-400 font-medium"
                        : "text-neutral-400"
                    }
                  >
                    {agent.error_count_last_hour ?? 0}
                  </span>
                </td>
                <td className="py-3 px-3 text-neutral-400">{timeAgo(agent.last_seen)}</td>
                <td className="py-3 px-3">
                  <span className="text-neutral-500 font-mono text-xs">
                    {agent.config_hash?.slice(0, 8) ?? "-"}
                  </span>
                </td>
                <td className="py-3 px-3">
                  <button
                    onClick={() => openDiagnostics(agent)}
                    className="text-green-400 hover:text-green-300 text-xs"
                  >
                    Details
                  </button>
                </td>
              </tr>
            ))}
            {sortedAgents.length === 0 && (
              <tr>
                <td colSpan={10} className="py-8 text-center text-neutral-500">
                  {loading ? "Loading agents..." : "No agents found"}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );

  // ── Tab: Error Timeline ──────────────────────────────────────────────

  const renderErrors = () => (
    <div className="space-y-4">
      <div className="flex gap-2 items-center">
        <select
          value={errorTypeFilter}
          onChange={(e) => setErrorTypeFilter(e.target.value)}
          className="bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-white"
        >
          <option value="">All Types</option>
          <option value="collection">Collection</option>
          <option value="upload">Upload</option>
          <option value="scheduler">Scheduler</option>
          <option value="power_control">Power Control</option>
          <option value="attribution">Attribution</option>
          <option value="auto_tuner">Auto-Tuner</option>
          <option value="unhandled">Unhandled</option>
        </select>
        <span className="text-neutral-500 text-sm ml-2">
          {errorsTotal} total errors
        </span>
      </div>

      <div className="space-y-2">
        {errors.map((err) => (
          <div
            key={err.id}
            className="bg-neutral-900 border border-neutral-800 rounded-lg p-4"
          >
            <div className="flex items-center gap-3 text-sm">
              <span
                className={`font-mono font-medium ${
                  ERROR_TYPE_COLORS[err.error_type] ?? "text-neutral-400"
                }`}
              >
                {err.error_type}
              </span>
              <span className="text-neutral-500">{err.hostname}</span>
              {err.gpu_index !== null && (
                <span className="text-neutral-500">GPU {err.gpu_index}</span>
              )}
              <span className="text-neutral-600 ml-auto">{timeAgo(err.created_at)}</span>
            </div>
            <p className="text-neutral-300 text-sm mt-2 font-mono">
              {err.error_message}
            </p>
            {err.stack_trace && (
              <>
                <button
                  onClick={() =>
                    setExpandedError(expandedError === err.id ? null : err.id)
                  }
                  className="text-xs text-neutral-500 hover:text-neutral-300 mt-2"
                >
                  {expandedError === err.id ? "Hide" : "Show"} stack trace
                </button>
                {expandedError === err.id && (
                  <pre className="mt-2 text-xs text-neutral-500 bg-neutral-950 rounded p-3 overflow-x-auto max-h-64">
                    {err.stack_trace}
                  </pre>
                )}
              </>
            )}
          </div>
        ))}
        {errors.length === 0 && (
          <div className="text-center text-neutral-500 py-12">
            {loading ? "Loading errors..." : "No errors recorded"}
          </div>
        )}
      </div>
    </div>
  );

  // ── Tab: Alerts ──────────────────────────────────────────────────────

  const renderAlerts = () => (
    <div className="space-y-4">
      <div className="flex gap-4 text-sm text-neutral-400">
        <span>
          Active: {alerts.filter((a) => !a.resolved).length}
        </span>
        <span>
          Resolved: {alerts.filter((a) => a.resolved).length}
        </span>
      </div>

      <div className="space-y-2">
        {alerts
          .filter((a) => !a.resolved)
          .map((alert) => (
            <div
              key={alert.id}
              className={`border rounded-lg p-4 ${
                SEVERITY_COLORS[alert.severity] ?? SEVERITY_COLORS.warning
              }`}
            >
              <div className="flex items-center gap-3">
                <span className="font-medium text-sm">
                  {ALERT_TYPE_LABELS[alert.alert_type] ?? alert.alert_type}
                </span>
                <span className="text-xs opacity-70 uppercase">{alert.severity}</span>
                <span className="text-xs opacity-60 ml-auto">
                  {timeAgo(alert.created_at)}
                </span>
              </div>
              <p className="text-sm mt-2 opacity-80">{alert.message}</p>
            </div>
          ))}
        {alerts.filter((a) => !a.resolved).length === 0 && (
          <div className="text-center text-neutral-500 py-12">
            {loading ? "Loading alerts..." : "No active alerts"}
          </div>
        )}
      </div>

      {/* Resolved alerts */}
      {alerts.filter((a) => a.resolved).length > 0 && (
        <div className="mt-8">
          <h3 className="text-sm text-neutral-500 mb-3">Recently Resolved</h3>
          <div className="space-y-2">
            {alerts
              .filter((a) => a.resolved)
              .slice(0, 20)
              .map((alert) => (
                <div
                  key={alert.id}
                  className="border border-neutral-800 rounded-lg p-3 opacity-50"
                >
                  <div className="flex items-center gap-3 text-sm">
                    <span className="text-neutral-400">
                      {ALERT_TYPE_LABELS[alert.alert_type] ?? alert.alert_type}
                    </span>
                    <span className="text-neutral-500">{alert.hostname}</span>
                    <span className="text-neutral-600 ml-auto">
                      Resolved {alert.resolved_at ? timeAgo(alert.resolved_at) : ""}
                    </span>
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );

  // ── Tab: Diagnostics (per-agent drilldown) ───────────────────────────

  const renderDiagnostics = () => {
    if (!selectedAgent) {
      return (
        <div className="text-center text-neutral-500 py-12">
          Select an agent from the Fleet tab to view diagnostics
        </div>
      );
    }

    const a = selectedAgent;

    return (
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setTab("fleet")}
            className="text-sm text-neutral-400 hover:text-white"
          >
            ← Back to Fleet
          </button>
          <span
            className={`inline-block w-2.5 h-2.5 rounded-full ${
              a.status === "online" ? "bg-green-500" : "bg-red-500"
            }`}
          />
          <h2 className="text-lg font-bold text-white">{a.hostname}</h2>
        </div>

        {/* Info grid */}
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4 space-y-2">
            <h3 className="text-xs text-neutral-500 uppercase tracking-wider">Agent</h3>
            <div className="text-sm space-y-1">
              <div className="flex justify-between">
                <span className="text-neutral-400">Version</span>
                <span className="text-white">v{a.agent_version}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-400">Mode</span>
                <span className="text-white">{a.agent_mode ?? "normal"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-400">Uptime</span>
                <span className="text-white">{uptimeStr(a.uptime_sec)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-400">Last Seen</span>
                <span className="text-white">{timeAgo(a.last_seen)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-400">Config Hash</span>
                <span className="text-white font-mono text-xs">{a.config_hash}</span>
              </div>
            </div>
          </div>

          <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4 space-y-2">
            <h3 className="text-xs text-neutral-500 uppercase tracking-wider">Hardware</h3>
            <div className="text-sm space-y-1">
              <div className="flex justify-between">
                <span className="text-neutral-400">GPUs</span>
                <span className="text-white">{a.gpu_count}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-400">Backend</span>
                <span className="text-white">{a.gpu_backend ?? "-"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-400">GPU Model</span>
                <span className="text-white">{a.gpu_names?.[0] ?? "-"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-400">Scheduler</span>
                <span className="text-white">{a.scheduler}</span>
              </div>
            </div>
          </div>

          <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4 space-y-2">
            <h3 className="text-xs text-neutral-500 uppercase tracking-wider">System</h3>
            <div className="text-sm space-y-1">
              <div className="flex justify-between">
                <span className="text-neutral-400">OS</span>
                <span className="text-white">{a.os_info ?? "-"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-400">Python</span>
                <span className="text-white">{a.python_version ?? "-"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-400">Machine ID</span>
                <span className="text-white font-mono text-xs">
                  {a.machine_id?.slice(0, 12) ?? "-"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-400">Cluster</span>
                <span className="text-white">{a.cluster_tag || "-"}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Error stats */}
        <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4">
          <h3 className="text-xs text-neutral-500 uppercase tracking-wider mb-3">
            Error Summary
          </h3>
          <div className="flex gap-8 text-sm">
            <div>
              <span className="text-neutral-400">Total errors: </span>
              <span className="text-white font-medium">{a.error_count_total ?? 0}</span>
            </div>
            <div>
              <span className="text-neutral-400">Last hour: </span>
              <span
                className={
                  (a.error_count_last_hour ?? 0) > 10
                    ? "text-red-400 font-medium"
                    : "text-white"
                }
              >
                {a.error_count_last_hour ?? 0}
              </span>
            </div>
            {a.last_error_message && (
              <div>
                <span className="text-neutral-400">Last error: </span>
                <span className="text-orange-400 font-mono text-xs">
                  {a.last_error_message.slice(0, 80)}
                  {a.last_error_at && ` (${timeAgo(a.last_error_at)})`}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Recent errors for this agent */}
        <div>
          <h3 className="text-xs text-neutral-500 uppercase tracking-wider mb-3">
            Recent Errors
          </h3>
          <div className="space-y-2">
            {agentErrors.map((err) => (
              <div
                key={err.id}
                className="bg-neutral-900 border border-neutral-800 rounded-lg p-3"
              >
                <div className="flex items-center gap-3 text-sm">
                  <span
                    className={`font-mono font-medium ${
                      ERROR_TYPE_COLORS[err.error_type] ?? "text-neutral-400"
                    }`}
                  >
                    {err.error_type}
                  </span>
                  {err.gpu_index !== null && (
                    <span className="text-neutral-500">GPU {err.gpu_index}</span>
                  )}
                  <span className="text-neutral-600 ml-auto">
                    {timeAgo(err.created_at)}
                  </span>
                </div>
                <p className="text-neutral-300 text-xs mt-1 font-mono">
                  {err.error_message}
                </p>
                {err.stack_trace && (
                  <>
                    <button
                      onClick={() =>
                        setExpandedError(expandedError === err.id ? null : err.id)
                      }
                      className="text-xs text-neutral-500 hover:text-neutral-300 mt-1"
                    >
                      {expandedError === err.id ? "Hide" : "Show"} trace
                    </button>
                    {expandedError === err.id && (
                      <pre className="mt-2 text-xs text-neutral-500 bg-neutral-950 rounded p-2 overflow-x-auto max-h-48">
                        {err.stack_trace}
                      </pre>
                    )}
                  </>
                )}
              </div>
            ))}
            {agentErrors.length === 0 && (
              <div className="text-neutral-500 text-sm py-4">No errors for this agent</div>
            )}
          </div>
        </div>
      </div>
    );
  };

  // ── Render ────────────────────────────────────────────────────────────

  const TABS: { key: Tab; label: string }[] = [
    { key: "fleet", label: "Fleet Overview" },
    { key: "errors", label: "Error Timeline" },
    { key: "alerts", label: "Alerts" },
    { key: "diagnostics", label: "Diagnostics" },
  ];

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <h1 className="text-2xl font-bold text-white mb-6">Agent Health</h1>

      {/* Tab bar */}
      <div className="flex gap-1 mb-6 border-b border-neutral-800">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              tab === t.key
                ? "border-green-500 text-green-400"
                : "border-transparent text-neutral-400 hover:text-white"
            }`}
          >
            {t.label}
            {t.key === "alerts" && alerts.filter((a) => !a.resolved).length > 0 && (
              <span className="ml-2 bg-red-600 text-white text-xs rounded-full px-1.5 py-0.5">
                {alerts.filter((a) => !a.resolved).length}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === "fleet" && renderFleet()}
      {tab === "errors" && renderErrors()}
      {tab === "alerts" && renderAlerts()}
      {tab === "diagnostics" && renderDiagnostics()}
    </div>
  );
}
