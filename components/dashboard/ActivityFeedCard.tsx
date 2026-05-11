"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";

interface ActivityEvent {
  id: string;
  action: string;
  timestamp: string;
  title: string | null;
  source: string | null;
}

const ACTION_DOTS: Record<string, { color: string; label: string }> = {
  created: { color: "bg-blue-400", label: "New" },
  approved: { color: "bg-green-400", label: "Approved" },
  applied: { color: "bg-emerald-400", label: "Applied" },
  rejected: { color: "bg-neutral-500", label: "Rejected" },
  rolled_back: { color: "bg-orange-400", label: "Rolled back" },
  expired: { color: "bg-neutral-600", label: "Expired" },
  command_applied: { color: "bg-green-400", label: "Executed" },
  command_failed: { color: "bg-red-400", label: "Failed" },
};

const SOURCE_LABELS: Record<string, string> = {
  auto_tuner: "Auto-Tuner",
  workload_analyzer: "Workload",
  carbon_scheduler: "Carbon",
  swarm_policy: "Swarm",
};

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  if (diff < 60_000) return `${Math.floor(diff / 1000)}s ago`;
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return `${Math.floor(diff / 86_400_000)}d ago`;
}

export default function ActivityFeedCard() {
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchActivity = useCallback(async () => {
    try {
      const res = await fetch("/api/dashboard/activity?limit=5");
      if (!res.ok) return;
      const data = await res.json();
      setEvents(data.events ?? []);
    } catch {
      // silent
    }
  }, []);

  useEffect(() => {
    fetchActivity().finally(() => setLoading(false));
    const interval = setInterval(fetchActivity, 15_000);
    return () => clearInterval(interval);
  }, [fetchActivity]);

  return (
    <div className="rounded-xl bg-neutral-900 border border-neutral-800 p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-neutral-400">
          Recent Activity
        </h3>
        <Link
          href="/dashboard/activity"
          className="text-xs text-green-400 hover:text-green-300 transition-colors"
        >
          View all &rarr;
        </Link>
      </div>

      {loading ? (
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="flex items-center gap-3 animate-pulse">
              <div className="w-2 h-2 rounded-full bg-neutral-700" />
              <div className="flex-1 h-3 bg-neutral-800 rounded" />
              <div className="w-10 h-3 bg-neutral-800 rounded" />
            </div>
          ))}
        </div>
      ) : events.length === 0 ? (
        <div className="text-neutral-600 text-xs text-center py-6">
          No activity yet
        </div>
      ) : (
        <div className="space-y-3">
          {events.map((event) => {
            const dot = ACTION_DOTS[event.action] ?? {
              color: "bg-neutral-500",
              label: event.action,
            };
            return (
              <div
                key={`${event.id}-${event.action}`}
                className="flex items-start gap-3 group"
              >
                <div
                  className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${dot.color}`}
                />
                <div className="flex-1 min-w-0">
                  <div className="text-xs text-neutral-300 truncate">
                    <span className="text-neutral-500">{dot.label}:</span>{" "}
                    {event.title ?? "Untitled"}
                  </div>
                  <div className="flex items-center gap-2 mt-0.5">
                    {event.source && (
                      <span className="text-[10px] text-neutral-600">
                        {SOURCE_LABELS[event.source] ?? event.source}
                      </span>
                    )}
                  </div>
                </div>
                <span className="text-[10px] text-neutral-600 whitespace-nowrap shrink-0 mt-0.5">
                  {timeAgo(event.timestamp)}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
