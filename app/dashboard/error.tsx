"use client";

import ErrorFallback from "@/components/ErrorFallback";

export default function DashboardError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="min-h-screen bg-neutral-950 p-6">
      <ErrorFallback
        error={error}
        reset={reset}
        title="Dashboard error"
      />
    </div>
  );
}
