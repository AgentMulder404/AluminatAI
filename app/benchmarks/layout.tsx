import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Green AI Index",
  description:
    "The world's first open leaderboard for AI energy efficiency. kWh per 1M tokens by model × GPU × framework.",
  openGraph: {
    title: "Green AI Index | AluminatiAI",
    description:
      "Community-sourced, anonymous rankings of LLM energy efficiency. kWh per 1M tokens. Compare models across GPUs and frameworks.",
    url: "https://aluminatiai.com/benchmarks",
    siteName: "AluminatiAI",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Green AI Index | AluminatiAI",
    description:
      "The world's first open leaderboard for AI energy efficiency. kWh per 1M tokens by model × GPU × framework.",
  },
};

export default function BenchmarksLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
