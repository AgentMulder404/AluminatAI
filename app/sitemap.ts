import { MetadataRoute } from "next";

const BASE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "https://nemulai.com";

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    { url: `${BASE_URL}/`, changeFrequency: "monthly", priority: 1.0 },
    { url: `${BASE_URL}/benchmarks`, changeFrequency: "weekly", priority: 0.8 },
    { url: `${BASE_URL}/carbon`, changeFrequency: "weekly", priority: 0.8 },
    { url: `${BASE_URL}/pricing`, changeFrequency: "monthly", priority: 0.9 },
    { url: `${BASE_URL}/enterprise`, changeFrequency: "monthly", priority: 0.8 },
    { url: `${BASE_URL}/dashboard`, changeFrequency: "weekly", priority: 0.7 },
  ];
}
