import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "NemulAI — GPU Cost Intelligence",
    template: "%s | NemulAI",
  },
  description:
    "Real-time GPU energy monitoring, cost attribution, and cloud savings comparison.",
  metadataBase: new URL("https://www.nemulai.com"),
  openGraph: {
    type: "website",
    siteName: "NemulAI",
    title: "NemulAI — GPU Cost Intelligence",
    description:
      "Real-time GPU energy monitoring, cost attribution, and cloud savings comparison.",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "NemulAI — GPU Cost Intelligence",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "NemulAI — GPU Cost Intelligence",
    description:
      "Real-time GPU energy monitoring, cost attribution, and cloud savings comparison.",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "NemulAI — GPU Cost Intelligence",
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
