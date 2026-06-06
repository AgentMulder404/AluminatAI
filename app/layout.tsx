import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "NemulAI — Cut Your GPU Bill. Automatically.",
    template: "%s | NemulAI",
  },
  description:
    "GPU cost intelligence that learns your workloads. Per-job attribution, waste detection, and self-learning optimization for AI teams.",
  metadataBase: new URL("https://www.nemulai.com"),
  openGraph: {
    type: "website",
    siteName: "NemulAI",
    title: "NemulAI — Cut Your GPU Bill. Automatically.",
    description:
      "GPU cost intelligence that learns your workloads. Per-job attribution, waste detection, and self-learning optimization for AI teams.",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "NemulAI — Cut Your GPU Bill. Automatically.",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "NemulAI — Cut Your GPU Bill. Automatically.",
    description:
      "GPU cost intelligence that learns your workloads. Per-job attribution, waste detection, and self-learning optimization for AI teams.",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "NemulAI — Cut Your GPU Bill. Automatically.",
      },
    ],
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-neutral-950 text-neutral-100 antialiased">
        {children}
      </body>
    </html>
  );
}
