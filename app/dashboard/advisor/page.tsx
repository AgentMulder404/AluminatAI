"use client";

import { useEffect, useState, useCallback } from "react";

// ── Types ──────────────────────────────────────────────────────────────────────

interface Recommendation {
  id: string;
  machine_id: string;
  hostname: string;
  gpu_index: number | null;
  gpu_name: string | null;
  source: string;
  category: string;
  priority: string;
  title: string;
  description: string;
  action: string;
  estimated_savings_pct: number;
  estimated_savings_usd: number;
  effort_score: number;
  status: string;
  approved_at: string | null;
  applied_at: string | null;
  actual_savings_pct: number | null;
  action_payload: Record<string, unknown>;
  created_at: string;
}

interface Summary {
  total: number;
  pending: number;
  applied: number;
  rejected: number;
  avg_savings_pct: number;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  if (diff < 60_000) return `${Math.floor(diff / 1000)}s ago`;
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return `${Math.floor(diff / 86_400_000)}d ago`;
}

const PRIORITY_STYLES: Record<string, string> = {
  P1: "bg-red-600/20 text-red-400 border-red-600/40",
  P2: "bg-yellow-600/20 text-yellow-400 border-yellow-600/40",
  P3: "bg-blue-600/20 text-blue-400 border-blue-600/40",
};

const STATUS_STYLES: Record<string, string> = {
  pending: "bg-neutral-700/50 text-neutral-300",
  approved: "bg-blue-600/20 text-blue-400",
  applied: "bg-green-600/20 text-green-400",
  rejected: "bg-neutral-800 text-neutral-500",
  expired: "bg-neutral-800 text-neutral-600",
  rolled_back: "bg-orange-600/20 text-orange-400",
};

const SOURCE_LABELS: Record<string, string> = {
  auto_tuner: "Auto-Tuner",
  workload_analyzer: "Workload Analyzer",
  carbon_scheduler: "Carbon Scheduler",
  swarm_policy: "Swarm Policy",
};

const CATEGORY_LABELS: Record<string, string> = {
  power_cap: "Power Cap",
  precision: "Precision",
  utilization: "Utilization",
  idle: "Idle GPU",
  carbon_schedule: "Carbon Schedule",
  gpu_match: "GPU Match",
  thermal: "Thermal",
  memory: "Memory",
};

const EFFORT_LABELS = ["", "Trivial", "Easy", "Moderate", "Complex", "Major"];

// ── Component ──────────────────────────────────────────────────────────────────

export default function AdvisorPage() {
  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [summary, setSummary] = useState<Summary>({
    total: 0, pending: 0, applied: 0, rejected: 0, avg_savings_pct: 0,
  });
  const [loading, setLoading] = useState(true);
  const [acting, setActing] = useState<string | null>(null);

  // Filters
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [sourceFilter, setSourceFilter] = useState<string>("");
  const [priorityFilter, setPriorityFilter] = useState<string>("");

  const fetchRecs = useCallback(async () => {
    const params = new URLSearchParams();
    if (statusFilter) params.set("status", statusFilter);
    if (sourceFilter) params.set("source", sourceFilter);
    if (priorityFilter) params.set("priority", priorityFilter);
    const res = await fetch(`/api/recommendations?${params}`);
    if (res.ok) {
      const data = await res.json();
      setRecs(data.recommendations ?? []);
      setSummary((prev) => data.summary ?? prev);
    }
  }, [statusFilter, sourceFilter, priorityFilter]);

  useEffect(() => {
    setLoading(true);
    fetchRecs().finally(() => setLoading(false));
    const interval = setInterval(fetchRecs, 15_000);
    return () => clearInterval(interval);
  }, [fetchRecs]);

  const handleApprove = async (id: string) => {
    setActing(id);
    const res = await fetch(`/api/recommendations/${id}/approve`, { method: "POST" });
    if (res.ok) {
      await fetchRecs();
    }
    setActing(null);
  };

  const handleReject = async (id: string) => {
    setActing(id);
    const res = await fetch(`/api/recommendations/${id}/reject`, { method: "POST" });
    if (res.ok) {
      await fetchRecs();
    }
    setActing(null);
  };

  const handleRollback = async (id: string) => {
    setActing(id);
    const res = await fetch(`/api/recommendations/${id}/rollback`, { method: "POST" });
    if (res.ok) {
      await fetchRecs();
    }
    setActing(null);
  };

  const pendingRecs = recs.filter((r) => r.status === "pending");
  const otherRecs = recs.filter((r) => r.status !== "pending");

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <h1 className="text-2xl font-bold text-white mb-2">Energy Advisor</h1>
      <p className="text-neutral-400 text-sm mb-6">
        AI-powered recommendations to reduce GPU energy waste and costs.
      </p>

      {/* Summary cards */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4">
          <div className="text-sm text-neutral-400">Avg Savings</div>
          <div className="text-3xl font-bold text-green-400 mt-1">
            {summary.avg_savings_pct > 0
              ? `~${summary.avg_savings_pct}%`
              : "—"}
          </div>
        </div>
        <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4">
          <div className="text-sm text-neutral-400">Pending</div>
          <div className="text-3xl font-bold text-yellow-400 mt-1">{summary.pending}</div>
        </div>
        <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4">
          <div className="text-sm text-neutral-400">Applied</div>
          <div className="text-3xl font-bold text-green-400 mt-1">{summary.applied}</div>
        </div>
        <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4">
          <div className="text-sm text-neutral-400">Rejected</div>
          <div className="text-3xl font-bold text-neutral-500 mt-1">{summary.rejected}</div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-2 mb-6">
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-white"
        >
          <option value="">All Status</option>
          <option value="pending">Pending</option>
          <option value="approved">Approved</option>
          <option value="applied">Applied</option>
          <option value="rejected">Rejected</option>
          <option value="rolled_back">Rolled Back</option>
        </select>
        <select
          value={sourceFilter}
          onChange={(e) => setSourceFilter(e.target.value)}
          className="bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-white"
        >
          <option value="">All Sources</option>
          <option value="auto_tuner">Auto-Tuner</option>
          <option value="workload_analyzer">Workload Analyzer</option>
          <option value="carbon_scheduler">Carbon Scheduler</option>
          <option value="swarm_policy">Swarm Policy</option>
        </select>
        <select
          value={priorityFilter}
          onChange={(e) => setPriorityFilter(e.target.value)}
          className="bg-neutral-900 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-white"
        >
          <option value="">All Priority</option>
          <option value="P1">P1 — Critical</option>
          <option value="P2">P2 — Recommended</option>
          <option value="P3">P3 — Nice to have</option>
        </select>
      </div>

      {/* Pending recommendations */}
      {pendingRecs.length > 0 && (
        <div className="space-y-3 mb-10">
          <h2 className="text-sm text-neutral-500 uppercase tracking-wider font-medium">
            Action Required ({pendingRecs.length})
          </h2>
          {pendingRecs.map((rec) => (
            <div
              key={rec.id}
              className="bg-neutral-900 border border-neutral-800 rounded-xl p-5"
            >
              <div className="flex items-start gap-3">
                {/* Priority badge */}
                <span
                  className={`px-2 py-0.5 rounded text-xs font-bold border ${
                    PRIORITY_STYLES[rec.priority] ?? PRIORITY_STYLES.P2
                  }`}
                >
                  {rec.priority}
                </span>

                <div className="flex-1 min-w-0">
                  <h3 className="text-white font-medium">{rec.title}</h3>
                  <p className="text-neutral-400 text-sm mt-1">{rec.description}</p>

                  {rec.action && (
                    <div className="mt-2 bg-neutral-950 rounded-lg px-3 py-2 text-sm font-mono text-green-400">
                      {rec.action}
                    </div>
                  )}

                  <div className="flex items-center gap-4 mt-3 text-xs text-neutral-500">
                    <span>{SOURCE_LABELS[rec.source] ?? rec.source}</span>
                    <span>{CATEGORY_LABELS[rec.category] ?? rec.category}</span>
                    {rec.gpu_name && <span>{rec.gpu_name}</span>}
                    {rec.gpu_index !== null && <span>GPU {rec.gpu_index}</span>}
                    <span>{rec.hostname}</span>
                    <span>{timeAgo(rec.created_at)}</span>
                  </div>
                </div>

                {/* Savings + effort */}
                <div className="text-right shrink-0 space-y-1">
                  {rec.estimated_savings_pct > 0 && (
                    <div className="text-green-400 font-bold text-lg">
                      ~{rec.estimated_savings_pct}%
                    </div>
                  )}
                  <div className="text-xs text-neutral-500">
                    Effort: {EFFORT_LABELS[rec.effort_score] ?? rec.effort_score}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex flex-col gap-2 shrink-0">
                  <button
                    onClick={() => handleApprove(rec.id)}
                    disabled={acting === rec.id}
                    className="px-4 py-1.5 bg-green-600 hover:bg-green-500 text-white text-sm rounded-lg font-medium disabled:opacity-50 transition-colors"
                  >
                    {acting === rec.id ? "..." : "Apply"}
                  </button>
                  <button
                    onClick={() => handleReject(rec.id)}
                    disabled={acting === rec.id}
                    className="px-4 py-1.5 bg-neutral-800 hover:bg-neutral-700 text-neutral-300 text-sm rounded-lg disabled:opacity-50 transition-colors"
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Empty state */}
      {!loading && recs.length === 0 && (
        <div className="text-center py-16">
          <div className="text-neutral-500 text-lg mb-2">No recommendations yet</div>
          <p className="text-neutral-600 text-sm max-w-md mx-auto">
            The agent analyzes your GPU workloads and surfaces optimization opportunities here.
            Make sure your agent is running with <code className="text-neutral-400">AUTO_TUNE_ENABLED=1</code>.
          </p>
        </div>
      )}

      {loading && recs.length === 0 && (
        <div className="text-center py-16 text-neutral-500">Loading recommendations...</div>
      )}

      {/* History */}
      {otherRecs.length > 0 && (
        <div className="space-y-2">
          <h2 className="text-sm text-neutral-500 uppercase tracking-wider font-medium">
            History ({otherRecs.length})
          </h2>
          {otherRecs.map((rec) => (
            <div
              key={rec.id}
              className={`bg-neutral-900/50 border border-neutral-800/50 rounded-lg p-4 ${
                rec.status === "rejected" ? "opacity-50" : ""
              }`}
            >
              <div className="flex items-center gap-3">
                <span
                  className={`px-2 py-0.5 rounded text-xs font-medium ${
                    STATUS_STYLES[rec.status] ?? STATUS_STYLES.pending
                  }`}
                >
                  {rec.status}
                </span>
                <span className="text-white text-sm">{rec.title}</span>
                <span className="text-neutral-600 text-xs ml-auto">
                  {timeAgo(rec.created_at)}
                </span>
                {rec.status === "applied" && rec.actual_savings_pct !== null && (
                  <span className="text-green-400 text-sm font-medium">
                    Actual: {rec.actual_savings_pct}%
                  </span>
                )}
                {rec.status === "applied" && rec.actual_savings_pct === null && (
                  <span className="text-neutral-500 text-xs">
                    est. {rec.estimated_savings_pct}%
                  </span>
                )}
                {rec.status === "applied" && !!(rec.action_payload as Record<string, unknown>)?.command && (
                  <button
                    onClick={() => handleRollback(rec.id)}
                    disabled={acting === rec.id}
                    className="px-3 py-1 bg-neutral-800 hover:bg-neutral-700 text-orange-400 text-xs rounded-lg disabled:opacity-50 transition-colors"
                  >
                    {acting === rec.id ? "..." : "Rollback"}
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
