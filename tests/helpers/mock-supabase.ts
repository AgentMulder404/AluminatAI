/**
 * Chainable mock matching the Supabase query builder pattern.
 * Each test calls setMockData() to define what queries return.
 */
import { vi } from "vitest";

interface MockResult {
  data: unknown;
  error: null | { message: string };
}

interface TableConfig {
  // Default data for any query on this table
  defaultData: unknown;
  defaultError: null | { message: string };
}

const _tables = new Map<string, TableConfig>();
let _insertError: null | { message: string } = null;
let _upsertError: null | { message: string } = null;

export function setMockData(
  table: string,
  data: unknown,
  error: null | { message: string } = null
) {
  _tables.set(table, { defaultData: data, defaultError: error });
}

export function setInsertError(error: null | { message: string }) {
  _insertError = error;
}

export function setUpsertError(error: null | { message: string }) {
  _upsertError = error;
}

export function resetMocks() {
  _tables.clear();
  _insertError = null;
  _upsertError = null;
}

function resolve(table: string): MockResult {
  const config = _tables.get(table);
  if (!config) return { data: null, error: null };
  return { data: config.defaultData, error: config.defaultError };
}

function createChain(table: string) {
  const chain: Record<string, unknown> = {};
  const methods = [
    "select",
    "eq",
    "neq",
    "gt",
    "gte",
    "lt",
    "lte",
    "not",
    "in",
    "is",
    "order",
    "limit",
    "range",
    "filter",
    "match",
    "textSearch",
  ];

  for (const method of methods) {
    chain[method] = vi.fn().mockReturnValue(chain);
  }

  // Terminal methods
  chain.single = vi.fn().mockImplementation(() => resolve(table));
  chain.maybeSingle = vi.fn().mockImplementation(() => resolve(table));
  chain.then = undefined; // Make it thenable via implicit promise resolution

  // insert / upsert / update / delete
  chain.insert = vi.fn().mockImplementation(() => ({
    ...chain,
    select: vi.fn().mockReturnValue({
      single: vi.fn().mockReturnValue({
        data: null,
        error: _insertError,
      }),
    }),
    error: _insertError,
    data: null,
  }));

  chain.upsert = vi.fn().mockImplementation(() => ({
    error: _upsertError,
    data: null,
  }));

  chain.update = vi.fn().mockReturnValue(chain);
  chain.delete = vi.fn().mockReturnValue(chain);

  // Make the chain itself resolve like a promise (for await supabase.from(...).select(...).eq(...))
  const resolvePromise = () => resolve(table);
  chain.then = (onfulfilled: (value: MockResult) => unknown) =>
    Promise.resolve(resolvePromise()).then(onfulfilled);

  return chain;
}

export function createMockSupabaseClient() {
  return {
    from: vi.fn().mockImplementation((table: string) => createChain(table)),
    auth: {
      getUser: vi.fn().mockResolvedValue({
        data: { user: null },
        error: { message: "Not authenticated" },
      }),
    },
  };
}

// Factory that vi.mock() can use
export const mockSupabaseClient = createMockSupabaseClient();
