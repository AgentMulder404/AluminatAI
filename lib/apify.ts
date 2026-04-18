const APIFY_BASE = "https://api.apify.com/v2";

function getToken(): string {
  const token = process.env.APIFY_API_TOKEN;
  if (!token) throw new Error("APIFY_API_TOKEN not configured");
  return token;
}

export async function startActorRun(
  actorId: string,
  input: Record<string, unknown>
): Promise<{ runId: string; datasetId: string }> {
  const token = getToken();
  const res = await fetch(
    `${APIFY_BASE}/acts/${actorId}/runs?token=${token}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }
  );
  const text = await res.text();
  if (!res.ok) {
    throw new Error(`Apify start failed (${res.status}): ${text}`);
  }
  if (!text || text.trim() === "") {
    throw new Error("Apify returned empty response when starting actor run");
  }
  const data = JSON.parse(text);
  return {
    runId: data.data.id,
    datasetId: data.data.defaultDatasetId,
  };
}

export async function getRunStatus(
  runId: string
): Promise<{ status: string; datasetId: string }> {
  const token = getToken();
  const res = await fetch(
    `${APIFY_BASE}/actor-runs/${runId}?token=${token}`
  );
  const statusText = await res.text();
  if (!res.ok) throw new Error(`Apify status failed (${res.status}): ${statusText}`);
  if (!statusText || statusText.trim() === "") {
    throw new Error("Apify returned empty response for run status");
  }
  const data = JSON.parse(statusText);
  return {
    status: data.data.status,
    datasetId: data.data.defaultDatasetId,
  };
}

export async function getDatasetItems<T = unknown>(
  datasetId: string,
  limit = 100,
  offset = 0
): Promise<T[]> {
  const token = getToken();
  const res = await fetch(
    `${APIFY_BASE}/datasets/${datasetId}/items?token=${token}&limit=${limit}&offset=${offset}`
  );
  if (!res.ok) throw new Error(`Apify dataset failed (${res.status})`);
  const text = await res.text();
  if (!text || text.trim() === "") return [];
  return JSON.parse(text);
}
