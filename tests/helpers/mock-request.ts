/**
 * Construct NextRequest-like objects for testing API routes.
 * Adds `nextUrl` property matching Next.js NextRequest behavior.
 */

function addNextUrl(req: Request): Request & { nextUrl: URL } {
  const r = req as Request & { nextUrl: URL };
  r.nextUrl = new URL(req.url);
  return r;
}

export function createRequest(
  url: string,
  options: {
    method?: string;
    headers?: Record<string, string>;
    body?: unknown;
  } = {}
) {
  const { method = "GET", headers = {}, body } = options;
  const init: RequestInit = {
    method,
    headers: new Headers(headers),
  };
  if (body !== undefined) {
    init.body = JSON.stringify(body);
  }
  const fullUrl = url.startsWith("http")
    ? url
    : `https://www.aluminatai.com${url}`;
  return addNextUrl(new Request(fullUrl, init));
}

export function createJsonRequest(
  path: string,
  body: unknown,
  headers: Record<string, string> = {}
) {
  return createRequest(`https://www.aluminatai.com${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body,
  });
}

export function createGetRequest(
  path: string,
  headers: Record<string, string> = {}
) {
  return createRequest(`https://www.aluminatai.com${path}`, {
    method: "GET",
    headers,
  });
}
