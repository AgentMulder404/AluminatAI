import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Green AI Index",
  description:
    "The world's first open leaderboard for AI energy efficiency. kWh per 1M tokens by model × GPU × framework.",
  openGraph: {
    title: "Green AI Index | NemulAI",
    description:
      "Community-sourced, anonymous rankings of LLM energy efficiency. kWh per 1M tokens. Compare models across GPUs and frameworks.",
    url: "https://nemulai.com/benchmarks",
    siteName: "NemulAI",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Green AI Index | NemulAI",
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
