"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

interface ClusterSummary {
  cluster_tag: string;
  machine_count: number;
  online_agent_count: number;
  gpu_count: number;
  total_kwh: number;
  avg_power_w: number;
  avg_utilization_pct: number;
}

interface ClusterFilterProps {
  onClusterChange: (tag: string | null) => void;
}

export default function ClusterFilter({ onClusterChange }: ClusterFilterProps) {
  const [clusters, setClusters] = useState<ClusterSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    fetch("/api/dashboard/clusters")
      .then((r) => (r.ok ? r.json() : []))
      .then((data) => setClusters(data as ClusterSummary[]))
      .catch(() => setClusters([]));
  }, []);

  useEffect(() => {
    const param = searchParams.get("cluster") ?? null;
    setSelected(param);
    onClusterChange(param);
  }, [searchParams, onClusterChange]);

  if (clusters.length === 0) return null;

  function select(tag: string | null) {
    const params = new URLSearchParams(searchParams.toString());
    if (tag) {
      params.set("cluster", tag);
    } else {
      params.delete("cluster");
    }
    router.replace(`?${params.toString()}`);
    setSelected(tag);
    onClusterChange(tag);
  }

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-xs text-neutral-500">Cluster:</span>
      <button
        onClick={() => select(null)}
        className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
          selected === null
            ? "bg-indigo-600 text-white"
            : "bg-neutral-800 text-neutral-300 hover:bg-neutral-700"
        }`}
      >
        All
      </button>
      {clusters.map((c) => (
        <button
          key={c.cluster_tag}
          onClick={() => select(c.cluster_tag)}
          className={`px-3 py-1 rounded-full text-xs font-medium transition-colors flex items-center gap-1.5 ${
            selected === c.cluster_tag
              ? "bg-indigo-600 text-white"
              : "bg-neutral-800 text-neutral-300 hover:bg-neutral-700"
          }`}
        >
          {c.cluster_tag}
          <span
            className={`opacity-75 ${
              selected === c.cluster_tag ? "text-indigo-200" : "text-neutral-500"
            }`}
          >
            {c.gpu_count} GPU
          </span>
        </button>
      ))}
    </div>
  );
}
