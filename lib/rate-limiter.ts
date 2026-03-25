/**
 * Sliding-window rate limiter with in-memory fallback.
 * Tries Upstash Redis when UPSTASH_REDIS_REST_URL + TOKEN are set;
 * falls back to in-process Map so deployments without Redis still work.
 */

export interface RateLimitResult {
  success: boolean;
  remaining: number;
  resetTime: number; // Unix ms
}

// ── In-memory fallback ────────────────────────────────────────────────────────

interface Record {
  count: number;
  resetTime: number;
}

const _map = new Map<string, Record>();

// Prune expired entries every 5 minutes
if (typeof setInterval !== "undefined") {
  setInterval(() => {
    const now = Date.now();
    for (const [k, v] of _map) {
      if (now > v.resetTime) _map.delete(k);
    }
  }, 5 * 60 * 1000);
}

function inMemory(
  key: string,
  maxRequests: number,
  windowMs: number
): RateLimitResult {
  const now = Date.now();
  const rec = _map.get(key);

  if (!rec || now > rec.resetTime) {
    _map.set(key, { count: 1, resetTime: now + windowMs });
    return { success: true, remaining: maxRequests - 1, resetTime: now + windowMs };
  }

  if (rec.count >= maxRequests) {
    return { success: false, remaining: 0, resetTime: rec.resetTime };
  }

  rec.count++;
  return {
    success: true,
    remaining: maxRequests - rec.count,
    resetTime: rec.resetTime,
  };
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Rate-limit a key (e.g. `ingest:<userId>`).
 * @param key         Unique identifier
 * @param maxRequests Max requests in window (default 100)
 * @param windowMs    Window in ms (default 60 s)
 */
export async function rateLimit(
  key: string,
  maxRequests = 100,
  windowMs = 60_000
): Promise<RateLimitResult> {
  // Attempt Redis if env vars present
  const url = process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN;

  if (url && token) {
    try {
      const { Redis } = await import("@upstash/redis");
      const { Ratelimit } = await import("@upstash/ratelimit");
      const redis = new Redis({ url, token });
      const limiter = new Ratelimit({
        redis,
        limiter: Ratelimit.slidingWindow(maxRequests, `${windowMs}ms`),
        analytics: false,
      });
      const { success, remaining, reset } = await limiter.limit(key);
      return { success, remaining, resetTime: Number(reset) };
    } catch {
      // Fall through to in-memory
    }
  }

  return inMemory(key, maxRequests, windowMs);
}

export function getRateLimitHeaders(
  result: RateLimitResult
): Record<string, string> {
  return {
    "X-RateLimit-Remaining": result.remaining.toString(),
    "X-RateLimit-Reset": Math.floor(result.resetTime / 1000).toString(),
  };
}
