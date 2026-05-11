"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";

// ── Types ────────────────────────────────────────────────────────────────────

interface ActivityEvent {
  id: string;
  event_type: string;
  action: string;
  timestamp: string;
  title: string | null;
  hostname: string | null;
  machine_id: string | null;
  gpu_name: string | null;
  source: string | null;
  category: string | null;
  priority: string | null;
  estimated_savings_pct: number | null;
  metadata: Record<string, unknown>;
}

// ── Event Config ─────────────────────────────────────────────────────────────

const EVENT_CONFIG: Record<
  string,
  { icon: string; color: string; bg: string; label: string }
> = {
  created: {
    icon: "+",
    color: "text-blue-400",
    bg: "bg-blue-400/10",
    label: "New recommendation",
  },
  approved: {
    icon: "✓",
    color: "text-green-400",
    bg: "bg-green-400/10",
    label: "Approved",
  },
  applied: {
    icon: "⚡",
    color: "text-emerald-400",
    bg: "bg-emerald-400/10",
    label: "Applied",
  },
  rejected: {
    icon: "✗",
    color: "text-neutral-500",
    bg: "bg-neutral-500/10",
    label: "Rejected",
  },
  rolled_back: {
    icon: "↩",
    color: "text-orange-400",
    bg: "bg-orange-400/10",
    label: "Rolled back",
  },
  expired: {
    icon: "⏱",
    color: "text-neutral-600",
    bg: "bg-neutral-600/10",
    label: "Expired",
  },
  feedback_recorded: {
    icon: "✉",
    color: "text-yellow-400",
    bg: "bg-yellow-400/10",
    label: "Feedback",
  },
  command_applied: {
    icon: "▶",
    color: "text-green-400",
    bg: "bg-green-400/10",
    label: "Command executed",
  },
  command_failed: {
    icon: "!",
    color: "text-red-400",
    bg: "bg-red-400/10",
    label: "Command failed",
  },
};

const SOURCE_LABELS: Record<string, string> = {
  auto_tuner: "Auto-Tuner",
  workload_analyzer: "Workload",
  carbon_scheduler: "Carbon",
  swarm_policy: "Swarm",
};

const PRIORITY_STYLES: Record<string, string> = {
  P1: "text-red-400 bg-red-400/10",
  P2: "text-yellow-400 bg-yellow-400/10",
  P3: "text-blue-400 bg-blue-400/10",
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  if (diff < 60_000) return `${Math.floor(diff / 1000)}s ago`;
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return `${Math.floor(diff / 86_400_000)}d ago`;
}

// ── Component ────────────────────────────────────────────────────────────────

export default function ActivityPage() {
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [hasMore, setHasMore] = useState(false);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loadingMore, setLoadingMore] = useState(false);

  const [actionFilter, setActionFilter] = useState("");
  const [sourceFilter, setSourceFilter] = useState("");
  const [priorityFilter, setPriorityFilter] = useState("");

  const buildUrl = useCallback(
    (off: number) => {
      const params = new URLSearchParams();
      params.set("limit", "50");
      params.set("offset", String(off));
      if (actionFilter) params.set("action", actionFilter);
      if (sourceFilter) params.set("source", sourceFilter);
      if (priorityFilter) params.set("priority", priorityFilter);
      return `/api/dashboard/activity?${params}`;
    },
    [actionFilter, sourceFilter, priorityFilter]
  );

  const fetchEvents = useCallback(
    async (append = false) => {
      try {
        const off = append ? offset : 0;
        const res = await fetch(buildUrl(off));
        if (!res.ok) return;
        const data = await res.json();
        if (append) {
          setEvents((prev) => [...prev, ...data.events]);
        } else {
          setEvents(data.events);
        }
        setTotal(data.total);
        setHasMore(data.has_more);
      } catch {
        // silent
      }
    },
    [buildUrl, offset]
  );

  useEffect(() => {
    setLoading(true);
    setOffset(0);
    fetchEvents().finally(() => setLoading(false));
  }, [actionFilter, sourceFilter, priorityFilter]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const interval = setInterval(() => fetchEvents(), 15_000);
    return () => clearInterval(interval);
  }, [fetchEvents]);

  const loadMore = async () => {
    const newOffset = offset + 50;
    setOffset(newOffset);
    setLoadingMore(true);
    const res = await fetch(buildUrl(newOffset));
    if (res.ok) {
      const data = await res.json();
      setEvents((prev) => [...prev, ...data.events]);
      setHasMore(data.has_more);
      setTotal(data.total);
    }
    setLoadingMore(false);
  };

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <div className="mb-6">
        <Link
          href="/dashboard"
          className="text-sm text-neutral-500 hover:text-neutral-300 transition-colors"
        >
          &larr; Dashboard
        </Link>
        <h1 className="text-2xl font-bold text-white mt-2">Activity Feed</h1>
        <p className="text-neutral-400 text-sm mt-1">
          Real-time timeline of recommendations, commands, and optimizations.
          {total > 0 && (
            <span className="text-neutral-500 ml-2">{total} events</span>
          )}
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 mb-6">
        <select
          value={actionFilter}
          onChange={(e) => setActionFilter(e.target.value)}
          className="bg-neutral-900 border border-neutral-800 rounded-lg px-3 py-2 text-sm text-neutral-300 focus:outline-none focus:border-neutral-600"
        >
          <option value="">All actions</option>
          <option value="created">Created</option>
          <option value="approved">Approved</option>
          <option value="applied">Applied</option>
          <option value="rejected">Rejected</option>
          <option value="rolled_back">Rolled back</option>
          <option value="expired">Expired</option>
          <option value="command_applied">Command executed</option>
          <option value="command_failed">Command failed</option>
        </select>

        <select
          value={sourceFilter}
          onChange={(e) => setSourceFilter(e.target.value)}
          className="bg-neutral-900 border border-neutral-800 rounded-lg px-3 py-2 text-sm text-neutral-300 focus:outline-none focus:border-neutral-600"
        >
          <option value="">All sources</option>
          <option value="auto_tuner">Auto-Tuner</option>
          <option value="workload_analyzer">Workload</option>
          <option value="carbon_scheduler">Carbon</option>
          <option value="swarm_policy">Swarm</option>
        </select>

        <select
          value={priorityFilter}
          onChange={(e) => setPriorityFilter(e.target.value)}
          className="bg-neutral-900 border border-neutral-800 rounded-lg px-3 py-2 text-sm text-neutral-300 focus:outline-none focus:border-neutral-600"
        >
          <option value="">All priorities</option>
          <option value="P1">P1 (Critical)</option>
          <option value="P2">P2 (Medium)</option>
          <option value="P3">P3 (Low)</option>
        </select>
      </div>

      {/* Timeline */}
      {loading ? (
        <div className="text-neutral-500 text-sm py-12 text-center">
          Loading activity...
        </div>
      ) : events.length === 0 ? (
        <div className="text-center py-16">
          <div className="text-neutral-600 text-4xl mb-3">~</div>
          <div className="text-neutral-500 text-sm">No activity yet.</div>
          <div className="text-neutral-600 text-xs mt-1">
            Events will appear here as the agent generates recommendations and
            executes commands.
          </div>
        </div>
      ) : (
        <div className="relative pl-8">
          {/* Vertical line */}
          <div className="absolute left-3 top-2 bottom-2 w-px bg-neutral-800" />

          <div className="space-y-1">
            {events.map((event) => {
              const config = EVENT_CONFIG[event.action] ?? {
                icon: "?",
                color: "text-neutral-500",
                bg: "bg-neutral-500/10",
                label: event.action,
              };

              return (
                <div key={`${event.id}-${event.action}`} className="relative group">
                  {/* Dot on the timeline */}
                  <div
                    className={`absolute -left-8 top-4 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${config.color} ${config.bg} border border-neutral-800`}
                  >
                    {config.icon}
                  </div>

                  {/* Event card */}
                  <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-4 hover:border-neutral-700 transition-colors">
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span
                            className={`text-sm font-medium ${config.color}`}
                          >
                            {config.label}
                          </span>
                          {event.source && (
                            <span className="text-xs px-2 py-0.5 rounded-full bg-neutral-800 text-neutral-400">
                              {SOURCE_LABELS[event.source] ?? event.source}
                            </span>
                          )}
                          {event.priority && (
                            <span
                              className={`text-xs px-2 py-0.5 rounded-full ${
                                PRIORITY_STYLES[event.priority] ??
                                "text-neutral-400 bg-neutral-800"
                              }`}
                            >
                              {event.priority}
                            </span>
                          )}
                        </div>

                        {event.title && (
                          <div className="text-sm text-neutral-300 mt-1 truncate">
                            {event.title}
                          </div>
                        )}

                        <div className="flex items-center gap-3 mt-2 text-xs text-neutral-500">
                          {event.hostname && (
                            <span className="font-mono">{event.hostname}</span>
                          )}
                          {event.gpu_name && <span>{event.gpu_name}</span>}
                          {event.estimated_savings_pct != null &&
                            event.estimated_savings_pct > 0 && (
                              <span className="text-green-500">
                                ~{event.estimated_savings_pct}% savings
                              </span>
                            )}
                        </div>
                      </div>

                      <div className="text-xs text-neutral-600 whitespace-nowrap shrink-0">
                        {timeAgo(event.timestamp)}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Load more */}
          {hasMore && (
            <div className="mt-6 text-center">
              <button
                onClick={loadMore}
                disabled={loadingMore}
                className="text-sm text-neutral-400 hover:text-white bg-neutral-900 border border-neutral-800 rounded-lg px-4 py-2 transition-colors disabled:opacity-50"
              >
                {loadingMore ? "Loading..." : "Load more"}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
