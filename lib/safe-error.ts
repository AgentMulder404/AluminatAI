export function safeError(error: unknown): string {
  console.error("[API Error]", error);
  return "Internal server error";
}
