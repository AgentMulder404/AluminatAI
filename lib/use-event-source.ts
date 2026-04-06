"use client";

import { useState, useEffect, useRef, useCallback } from "react";

interface UseEventSourceOptions {
  fallbackInterval?: number; // Polling fallback interval in ms (default: 60000)
  enabled?: boolean; // Whether to connect (default: true)
}

interface UseEventSourceResult<T> {
  data: T | null;
  connected: boolean;
  error: Error | null;
}

export function useEventSource<T>(
  url: string,
  eventName: string,
  opts: UseEventSourceOptions = {}
): UseEventSourceResult<T> {
  const { fallbackInterval = 60000, enabled = true } = opts;
  const [data, setData] = useState<T | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const retriesRef = useRef(0);
  const esRef = useRef<EventSource | null>(null);
  const fallbackRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const maxRetries = 3;

  const cleanup = useCallback(() => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    if (fallbackRef.current) {
      clearInterval(fallbackRef.current);
      fallbackRef.current = null;
    }
  }, []);

  const startFallbackPolling = useCallback(() => {
    if (fallbackRef.current) return;
    fallbackRef.current = setInterval(async () => {
      try {
        const res = await fetch(url);
        if (res.ok) {
          const reader = res.body?.getReader();
          if (reader) {
            const { value } = await reader.read();
            if (value) {
              const text = new TextDecoder().decode(value);
              // Try to parse SSE data from the response
              const dataMatch = text.match(
                new RegExp(`event: ${eventName}\\ndata: (.+)\\n`)
              );
              if (dataMatch) {
                try {
                  setData(JSON.parse(dataMatch[1]));
                } catch {
                  // Ignore parse errors
                }
              }
            }
            reader.cancel();
          }
        }
      } catch {
        // Ignore polling errors
      }
    }, fallbackInterval);
  }, [url, eventName, fallbackInterval]);

  useEffect(() => {
    if (!enabled) {
      cleanup();
      return;
    }

    const connect = () => {
      cleanup();

      try {
        const es = new EventSource(url);
        esRef.current = es;

        es.addEventListener(eventName, (event) => {
          try {
            const parsed = JSON.parse(event.data);
            setData(parsed);
            setError(null);
          } catch {
            // Ignore parse errors
          }
        });

        es.onopen = () => {
          setConnected(true);
          setError(null);
          retriesRef.current = 0;
        };

        es.onerror = () => {
          setConnected(false);
          es.close();
          esRef.current = null;

          retriesRef.current++;

          if (retriesRef.current <= maxRetries) {
            // Exponential backoff: 1s, 2s, 4s
            const delay = Math.min(1000 * Math.pow(2, retriesRef.current - 1), 30000);
            setTimeout(connect, delay);
          } else {
            // Fall back to polling
            setError(new Error("SSE connection failed, falling back to polling"));
            startFallbackPolling();
          }
        };
      } catch (e) {
        setError(e instanceof Error ? e : new Error("Failed to create EventSource"));
        startFallbackPolling();
      }
    };

    connect();

    return cleanup;
  }, [url, eventName, enabled, cleanup, startFallbackPolling]);

  return { data, connected, error };
}
