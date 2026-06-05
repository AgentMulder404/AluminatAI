// API key authentication for agent-facing endpoints.
// Validates keys against the users table (populated by nemulai-landing auth).

import { createSupabaseServerClient } from "./supabase-client";

export interface ApiAuthResult {
  valid: boolean;
  userId: string;
}

/**
 * Validate a raw API key string (extracted from X-API-Key header by caller).
 * Returns { valid: true, userId } on success, { valid: false, userId: "" } on failure.
 */
export async function validateApiKey(apiKey: string): Promise<ApiAuthResult> {
  if (!apiKey || !apiKey.startsWith("alum_")) {
    return { valid: false, userId: "" };
  }

  try {
    const supabase = createSupabaseServerClient();
    const { data: user, error } = await supabase
      .from("users")
      .select("id, trial_ends_at")
      .eq("api_key", apiKey)
      .single();

    if (error || !user) {
      return { valid: false, userId: "" };
    }

    // Reject expired trials
    if (user.trial_ends_at && new Date(user.trial_ends_at) < new Date()) {
      return { valid: false, userId: "" };
    }

    return { valid: true, userId: user.id };
  } catch {
    return { valid: false, userId: "" };
  }
}
