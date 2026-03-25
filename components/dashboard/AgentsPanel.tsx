"use client";

import { useEffect, useState } from "react";

interface AgentRow {
  hostname: string;
  machine_id: string | null;
  cluster_tag: string;
  location_hint: string;
  gpu_count: number;
  gpu_names: string[] | null;
  agent_version: string;
  scheduler: string;
  last_seen: string;
  is_online: boolean;
}

function timeAgo(iso: string): string {
  const diffMs = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diffMs / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

function groupByCluster(agents: AgentRow[]): Map<string, AgentRow[]> {
  const map = new Map<string, AgentRow[]>();
  for (const agent of agents) {
    const key = agent.cluster_tag || "Default";
    const list = map.get(key) ?? [];
    list.push(agent);
    map.set(key, list);
  }
  return map;
}

export default function AgentsPanel() {
  const [agents, setAgents] = useState<AgentRow[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/agent/agents")
      .then((r) => (r.ok ? r.json() : []))
      .then((data) => setAgents(data as AgentRow[]))
      .catch(() => setAgents([]))
      .finally(() => setLoading(false));
  }, []);

  if (loading || agents.length === 0) return null;

  const groups = groupByCluster(agents);

  return (
    <div className="border border-neutral-800 bg-neutral-950 rounded-lg p-4">
      <h2 className="text-sm font-semibold text-neutral-300 mb-3">Agents</h2>
      <div className="space-y-4">
        {Array.from(groups.entries()).map(([cluster, rows]) => (
          <div key={cluster}>
            {groups.size > 1 && (
              <p className="text-xs text-neutral-500 font-medium uppercase tracking-wide mb-1">
                {cluster}
              </p>
            )}
            <table className="w-full text-xs text-neutral-300">
              <thead>
                <tr className="text-neutral-500 border-b border-neutral-800">
                  <th className="text-left pb-1 pr-3 font-normal">Host</th>
                  <th className="text-right pb-1 pr-3 font-normal">GPUs</th>
                  <th className="text-left pb-1 pr-3 font-normal">Model</th>
                  <th className="text-right pb-1 pr-3 font-normal">Version</th>
                  <th className="text-right pb-1 font-normal">Last seen</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((a) => (
                  <tr
                    key={a.machine_id ?? a.hostname}
                    className="border-b border-neutral-900 last:border-0"
                  >
                    <td className="py-1 pr-3 flex items-center gap-1.5">
                      <span
                        className={`inline-block h-1.5 w-1.5 rounded-full flex-shrink-0 ${
                          a.is_online ? "bg-green-400" : "bg-red-500"
                        }`}
                      />
                      {a.hostname}
                    </td>
                    <td className="py-1 pr-3 text-right">{a.gpu_count}</td>
                    <td className="py-1 pr-3 text-neutral-400 truncate max-w-[160px]">
                      {a.gpu_names?.slice(0, 2).join(", ") ?? "—"}
                    </td>
                    <td className="py-1 pr-3 text-right text-neutral-500">
                      v{a.agent_version}
                    </td>
                    <td className="py-1 text-right text-neutral-500">
                      {timeAgo(a.last_seen)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </div>
  );
}
