"use client";

import { useEffect, useState, useCallback } from "react";

// ── Types ──────────────────────────────────────────────────────────────────────

interface Leader {
  user_id: string;
  cluster_tag: string;
  machine_id: string;
  lease_token: string;
  acquired_at: string;
  expires_at: string;
  is_active: boolean;
  expires_in_s: number;
}

interface ClusterSummary {
  cluster_tag: string;
  machines: number;
  total_gpus: number;
  hostnames: string[];
}

interface RecommendationStats {
  total_24h: number;
  by_status: Record<string, number>;
  by_source: Record<string, number>;
  by_priority: Record<string, number>;
  avg_applied_savings_pct: number;
  funnel: {
    pending: number;
    approved: number;
    applied: number;
    rejected: number;
    rolled_back: number;
  };
}

interface CommandEntry {
  status: string;
  command_type: string;
  machine_id: string;
  created_at: string;
  completed_at: string | null;
}

interface CommandStats {
  total_24h: number;
  by_status: Record<string, number>;
  by_type: Record<string, number>;
  recent: CommandEntry[];
  avg_dispatch_latency_s: number | null;
  avg_exec_latency_s: number | null;
}

interface SwarmData {
  leaders: Leader[];
  fleet: {
    total_machines: number;
    total_gpus: number;
    clusters: ClusterSummary[];
  };
  recommendations: RecommendationStats;
  commands: CommandStats;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  if (diff < 60_000) return `${Math.floor(diff / 1000)}s ago`;
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return `${Math.floor(diff / 86_400_000)}d ago`;
}

const CMD_STATUS_COLORS: Record<string, string> = {
  pending: "text-yellow-400",
  dispatched: "text-blue-400",
  applied: "text-green-400",
  failed: "text-red-400",
};

const REC_SOURCE_LABELS: Record<string, string> = {
  auto_tuner: "Auto-Tuner",
  workload_analyzer: "Workload",
  carbon_scheduler: "Carbon",
  swarm_policy: "Swarm",
};

// ── Component ──────────────────────────────────────────────────────────────────

export default function SwarmControlPanel() {
  const [data, setData] = useState<SwarmData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch("/api/admin/swarm-status");
      if (!res.ok) throw new Error();
      const json = await res.json();
      setData(json);
      setError(false);
    } catch {
      setError(true);
    }
  }, []);

  useEffect(() => {
    setLoading(true);
    fetchData().finally(() => setLoading(false));
    const interval = setInterval(fetchData, 15_000);
    return () => clearInterval(interval);
  }, [fetchData]);

  if (loading && !data) {
    return (
      <div className="p-8 max-w-6xl mx-auto">
        <h1 className="text-2xl font-bold text-white mb-6">Swarm Control Panel</h1>
        <div className="text-neutral-500">Loading...</div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-8 max-w-6xl mx-auto">
        <h1 className="text-2xl font-bold text-white mb-6">Swarm Control Panel</h1>
        <div className="text-red-400">Failed to load swarm status.</div>
      </div>
    );
  }

  const { leaders, fleet, recommendations: recs, commands: cmds } = data;

  const funnelTotal = recs.total_24h || 1;
  const funnelSteps = [
    { label: "Generated", count: recs.total_24h, color: "bg-neutral-600" },
    { label: "Pending", count: recs.funnel.pending, color: "bg-yellow-500" },
    { label: "Approved", count: recs.funnel.approved, color: "bg-blue-500" },
    { label: "Applied", count: recs.funnel.applied, color: "bg-green-500" },
    { label: "Rejected", count: recs.funnel.rejected, color: "bg-neutral-700" },
    { label: "Rolled Back", count: recs.funnel.rolled_back, color: "bg-orange-500" },
  ];

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Swarm Control Panel</h1>
        <p className="text-neutral-400 text-sm mt-1">
          Fleet-wide optimization engine — leader election, policies, and command pipeline.
        </p>
      </div>

      {/* ── Top Stats ──────────────────────────────────────────────────── */}
      <div className="grid grid-cols-5 gap-4">
        <StatCard label="Machines" value={fleet.total_machines} color="text-white" />
        <StatCard label="GPUs" value={fleet.total_gpus} color="text-white" />
        <StatCard label="Recs (24h)" value={recs.total_24h} color="text-blue-400" />
        <StatCard label="Applied" value={recs.funnel.applied} color="text-green-400" />
        <StatCard
          label="Avg Savings"
          value={recs.avg_applied_savings_pct > 0 ? `${recs.avg_applied_savings_pct}%` : "—"}
          color="text-green-400"
        />
      </div>

      {/* ── Leader Election ────────────────────────────────────────────── */}
      <Section title="Leader Election">
        {leaders.length === 0 ? (
          <div className="text-neutral-500 text-sm">No swarm leaders registered. Enable with SWARM_ENABLED=1.</div>
        ) : (
          <div className="space-y-2">
            {leaders.map((l) => (
              <div
                key={`${l.cluster_tag}-${l.machine_id}`}
                className="flex items-center gap-4 bg-neutral-900 border border-neutral-800 rounded-lg px-4 py-3"
              >
                <span
                  className={`w-2 h-2 rounded-full shrink-0 ${
                    l.is_active ? "bg-green-500" : "bg-neutral-600"
                  }`}
                />
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-white font-medium">
                    {l.machine_id}
                    <span className="text-neutral-500 ml-2 font-normal">
                      cluster: {l.cluster_tag || "(default)"}
                    </span>
                  </div>
                </div>
                <div className="text-xs text-neutral-500">
                  Acquired {timeAgo(l.acquired_at)}
                </div>
                {l.is_active ? (
                  <span className="text-xs text-green-400 bg-green-600/10 px-2 py-0.5 rounded">
                    Active — {l.expires_in_s}s left
                  </span>
                ) : (
                  <span className="text-xs text-neutral-500 bg-neutral-800 px-2 py-0.5 rounded">
                    Expired
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </Section>

      {/* ── Fleet Overview ─────────────────────────────────────────────── */}
      <Section title="Fleet by Cluster">
        {fleet.clusters.length === 0 ? (
          <div className="text-neutral-500 text-sm">No active agents.</div>
        ) : (
          <div className="grid grid-cols-2 gap-3">
            {fleet.clusters.map((c) => (
              <div
                key={c.cluster_tag}
                className="bg-neutral-900 border border-neutral-800 rounded-lg p-4"
              >
                <div className="text-sm text-white font-medium mb-2">{c.cluster_tag}</div>
                <div className="flex gap-4 text-xs text-neutral-400">
                  <span>{c.machines} machines</span>
                  <span>{c.total_gpus} GPUs</span>
                </div>
                <div className="mt-2 text-xs text-neutral-600 truncate">
                  {c.hostnames.slice(0, 5).join(", ")}
                  {c.hostnames.length > 5 && ` +${c.hostnames.length - 5} more`}
                </div>
              </div>
            ))}
          </div>
        )}
      </Section>

      {/* ── Policy / Recommendation Stats ──────────────────────────────── */}
      <Section title="Recommendation Pipeline (24h)">
        <div className="grid grid-cols-2 gap-6">
          {/* Funnel */}
          <div>
            <div className="text-xs text-neutral-500 uppercase tracking-wider mb-3">Funnel</div>
            <div className="space-y-2">
              {funnelSteps.map((step) => (
                <div key={step.label} className="flex items-center gap-3">
                  <div className="w-20 text-xs text-neutral-400 text-right">{step.label}</div>
                  <div className="flex-1 h-5 bg-neutral-900 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${step.color} rounded-full transition-all`}
                      style={{ width: `${Math.max(2, (step.count / funnelTotal) * 100)}%` }}
                    />
                  </div>
                  <div className="w-8 text-xs text-neutral-300 text-right">{step.count}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Breakdowns */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-xs text-neutral-500 uppercase tracking-wider mb-2">By Source</div>
              {Object.entries(recs.by_source).map(([src, count]) => (
                <div key={src} className="flex justify-between text-xs py-1">
                  <span className="text-neutral-400">{REC_SOURCE_LABELS[src] ?? src}</span>
                  <span className="text-white">{count}</span>
                </div>
              ))}
              {Object.keys(recs.by_source).length === 0 && (
                <div className="text-xs text-neutral-600">No data</div>
              )}
            </div>
            <div>
              <div className="text-xs text-neutral-500 uppercase tracking-wider mb-2">By Priority</div>
              {["P1", "P2", "P3"].map((p) => (
                <div key={p} className="flex justify-between text-xs py-1">
                  <span className={
                    p === "P1" ? "text-red-400" : p === "P2" ? "text-yellow-400" : "text-blue-400"
                  }>
                    {p}
                  </span>
                  <span className="text-white">{recs.by_priority[p] ?? 0}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Section>

      {/* ── Command Pipeline ───────────────────────────────────────────── */}
      <Section title="Command Pipeline (24h)">
        <div className="grid grid-cols-4 gap-4 mb-4">
          <MiniStat label="Total" value={cmds.total_24h} />
          <MiniStat
            label="Dispatch Latency"
            value={cmds.avg_dispatch_latency_s !== null ? `${cmds.avg_dispatch_latency_s}s` : "—"}
          />
          <MiniStat
            label="Exec Latency"
            value={cmds.avg_exec_latency_s !== null ? `${cmds.avg_exec_latency_s}s` : "—"}
          />
          <MiniStat
            label="Success Rate"
            value={
              cmds.total_24h > 0
                ? `${Math.round(((cmds.by_status["applied"] ?? 0) / cmds.total_24h) * 100)}%`
                : "—"
            }
          />
        </div>

        {/* Status breakdown */}
        <div className="flex gap-3 mb-4">
          {Object.entries(cmds.by_status).map(([status, count]) => (
            <span
              key={status}
              className={`text-xs px-2 py-1 rounded bg-neutral-900 border border-neutral-800 ${
                CMD_STATUS_COLORS[status] ?? "text-neutral-400"
              }`}
            >
              {status}: {count}
            </span>
          ))}
        </div>

        {/* Recent commands */}
        {cmds.recent.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-neutral-500 border-b border-neutral-800">
                  <th className="text-left py-2 pr-3">Status</th>
                  <th className="text-left py-2 pr-3">Type</th>
                  <th className="text-left py-2 pr-3">Machine</th>
                  <th className="text-left py-2 pr-3">Created</th>
                  <th className="text-left py-2">Completed</th>
                </tr>
              </thead>
              <tbody>
                {cmds.recent.map((c, i) => (
                  <tr key={i} className="border-b border-neutral-900">
                    <td className={`py-2 pr-3 ${CMD_STATUS_COLORS[c.status] ?? "text-neutral-400"}`}>
                      {c.status}
                    </td>
                    <td className="py-2 pr-3 text-neutral-300 font-mono">{c.command_type}</td>
                    <td className="py-2 pr-3 text-neutral-400 font-mono">{c.machine_id.slice(0, 12)}</td>
                    <td className="py-2 pr-3 text-neutral-500">{timeAgo(c.created_at)}</td>
                    <td className="py-2 text-neutral-500">
                      {c.completed_at ? timeAgo(c.completed_at) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>
    </div>
  );
}

// ── Shared Components ──────────────────────────────────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-neutral-950 border border-neutral-800 rounded-xl p-5">
      <h2 className="text-sm text-neutral-500 uppercase tracking-wider font-medium mb-4">{title}</h2>
      {children}
    </div>
  );
}

function StatCard({ label, value, color }: { label: string; value: string | number; color: string }) {
  return (
    <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4">
      <div className="text-xs text-neutral-400">{label}</div>
      <div className={`text-2xl font-bold mt-1 ${color}`}>{value}</div>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-3">
      <div className="text-xs text-neutral-500">{label}</div>
      <div className="text-sm font-medium text-white mt-0.5">{value}</div>
    </div>
  );
}
