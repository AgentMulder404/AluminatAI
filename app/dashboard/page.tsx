"use client";

import { Suspense, useCallback, useState } from "react";
import { useSearchParams } from "next/navigation";

import ClusterFilter from "@/components/dashboard/ClusterFilter";
import DateRangePicker from "@/components/dashboard/DateRangePicker";
import StatCard from "@/components/dashboard/StatCard";
import SavingsCard from "@/components/dashboard/SavingsCard";
import CostTrendChart from "@/components/dashboard/CostTrendChart";
import CostBreakdownPanel from "@/components/dashboard/CostBreakdownPanel";
import WasteAlerts from "@/components/dashboard/WasteAlerts";
import UtilizationHeatmap from "@/components/dashboard/UtilizationHeatmap";
import RecommendationsCard from "@/components/dashboard/RecommendationsCard";
import JobsCarbonTable from "@/components/dashboard/JobsCarbonTable";
import JobDetailDrawer from "@/components/dashboard/JobDetailDrawer";
import AgentsPanel from "@/components/dashboard/AgentsPanel";
import BenchmarkPercentileCard from "@/components/dashboard/BenchmarkPercentileCard";
import NotificationBell from "@/components/dashboard/NotificationBell";
import ActivityFeedCard from "@/components/dashboard/ActivityFeedCard";

function daysAgo(n: number): string {
  return new Date(Date.now() - n * 86400000).toISOString().slice(0, 10);
}

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
  const [dateRange, setDateRange] = useState({
    from: daysAgo(30),
    to: new Date().toISOString().slice(0, 10),
  });
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  const handleClusterChange = useCallback((tag: string | null) => {
    setSelectedCluster(tag);
  }, []);

  const clusterParam = selectedCluster
    ? `cluster_tag=${encodeURIComponent(selectedCluster)}`
    : "";

  const clusterTag = selectedCluster ?? undefined;

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 p-6 space-y-6">
      {/* Top bar: Cluster filter + Date range */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <ClusterFilter onClusterChange={handleClusterChange} />
        <div className="flex items-center gap-3">
          <NotificationBell />
          <DateRangePicker
          from={dateRange.from}
          to={dateRange.to}
          onChange={(from, to) => setDateRange({ from, to })}
        />
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <SavingsCard />
        <StatCard
          title="Today's Cost"
          fetchUrl={`/api/dashboard/today-cost?${clusterParam}`}
          valueKey="cost_usd"
          format="currency"
          streamEvent="metrics"
        />
        <StatCard
          title="Active GPUs"
          fetchUrl="/api/dashboard/clusters"
          valueKey="gpu_count"
          format="number"
        />
        <StatCard
          title="CO2 Emissions"
          fetchUrl={`/api/dashboard/today-cost?${clusterParam}`}
          valueKey="co2e_g"
          format="co2"
        />
      </div>

      {/* Cost Trend Chart */}
      <CostTrendChart
        clusterParam={clusterTag}
        from={dateRange.from}
        to={dateRange.to}
      />

      {/* Cost Breakdown + Waste Alerts (2-column) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CostBreakdownPanel
          clusterParam={clusterTag}
          from={dateRange.from}
          to={dateRange.to}
        />
        <WasteAlerts clusterParam={clusterTag} />
      </div>

      {/* Utilization Heatmap */}
      <UtilizationHeatmap clusterParam={clusterTag} />

      {/* Recommendations */}
      <RecommendationsCard clusterParam={clusterTag} />

      {/* Jobs Table — click a row to open detail drawer */}
      <JobsCarbonTable
        clusterParam={clusterParam}
        onJobClick={(jobId: string) => setSelectedJobId(jobId)}
      />

      {/* Agents + Activity + Benchmark (3-column) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <AgentsPanel />
        <ActivityFeedCard />
        <BenchmarkPercentileCard />
      </div>

      {/* Job Detail Drawer */}
      <JobDetailDrawer
        jobId={selectedJobId}
        onClose={() => setSelectedJobId(null)}
      />
    </div>
  );
}
