"use client";

import { Suspense, useCallback, useState } from "react";
import { useSearchParams } from "next/navigation";
import AgentsPanel from "@/components/dashboard/AgentsPanel";
import BenchmarkPercentileCard from "@/components/dashboard/BenchmarkPercentileCard";
import ClusterFilter from "@/components/dashboard/ClusterFilter";
import JobsCarbonTable from "@/components/dashboard/JobsCarbonTable";
// Existing dashboard components (adjust import paths as needed)
// import StatCard from "@/components/dashboard/StatCard";
// import UtilizationChart from "@/components/dashboard/UtilizationChart";
// import JobsTable from "@/components/dashboard/JobsTable";

export default function DashboardPage() {
  return (
    <Suspense>
      <DashboardInner />
    </Suspense>
  );
}

function DashboardInner() {
  const searchParams = useSearchParams();
  const [selectedCluster, setSelectedCluster] = useState<string | null>(
    searchParams.get("cluster") ?? null
  );

  const handleClusterChange = useCallback((tag: string | null) => {
    setSelectedCluster(tag);
  }, []);

  // Build cluster-scoped query param for child fetch URLs
  const clusterParam = selectedCluster
    ? `cluster_tag=${encodeURIComponent(selectedCluster)}`
    : "";

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 p-6 space-y-6">
      {/* Cluster filter — renders nothing for single-cluster users */}
      <ClusterFilter onClusterChange={handleClusterChange} />

      {/* KPI cards row — pass clusterParam to each card's fetch URL */}
      {/* <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard title="Today's Cost" fetchUrl={`/api/dashboard/today-cost?${clusterParam}`} />
        ...
      </div> */}

      {/* Utilization chart */}
      {/* <UtilizationChart fetchUrl={`/api/dashboard/utilization-chart?${clusterParam}`} /> */}

      {/* Benchmark percentile card */}
      <BenchmarkPercentileCard />

      {/* Agents panel — shows multi-cluster breakdown */}
      <AgentsPanel />

      {/* Jobs carbon table */}
      <JobsCarbonTable clusterParam={clusterParam} />
    </div>
  );
}
