// GET/POST/PATCH/DELETE /api/settings/export
// Manage S3/GCS export configurations. Cookie auth.

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { checkCountLimit } from "@/lib/plans";
import { encrypt } from "@/lib/crypto";
import { logAudit } from "@/lib/audit";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

import { safeError } from "@/lib/safe-error";
export const runtime = "edge";

async function authenticate() {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error,
  } = await cookieClient.auth.getUser();
  if (error || !user) return null;
  return user;
}

// GET — list export configs
export async function GET() {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`settings-export:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const supabase = createSupabaseServerClient();

  const { data, error } = await supabase
    .from("export_configs")
    .select(
      "id, provider, bucket, prefix, region, format, schedule, " +
        "last_export_at, is_active, created_at"
    )
    .eq("user_id", user.id)
    .order("created_at", { ascending: false });

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  return NextResponse.json({ configs: data ?? [] });
}

// POST — create an export config
export async function POST(req: NextRequest) {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`settings-export:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const body = await req.json();
  const {
    provider,
    bucket,
    prefix,
    region,
    format,
    schedule,
    access_key_id,
    secret_key,
    gcs_credentials,
  } = body;

  // Plan limit check
  const supabaseCount = createSupabaseServerClient();
  const { count: exportCount } = await supabaseCount
    .from("export_configs")
    .select("id", { count: "exact", head: true })
    .eq("user_id", user.id);
  const limitCheck = await checkCountLimit(user.id, "max_export_configs", exportCount ?? 0);
  if (!limitCheck.allowed) {
    return NextResponse.json({ error: limitCheck.reason, limit: limitCheck.limit }, { status: 403 });
  }

  if (!provider || !["s3", "gcs"].includes(provider)) {
    return NextResponse.json(
      { error: "provider must be 's3' or 'gcs'" },
      { status: 400 }
    );
  }

  if (!bucket || typeof bucket !== "string") {
    return NextResponse.json(
      { error: "bucket is required" },
      { status: 400 }
    );
  }

  if (provider === "s3" && (!access_key_id || !secret_key)) {
    return NextResponse.json(
      { error: "access_key_id and secret_key are required for S3" },
      { status: 400 }
    );
  }

  if (provider === "gcs" && !gcs_credentials) {
    return NextResponse.json(
      { error: "gcs_credentials (service account JSON) is required for GCS" },
      { status: 400 }
    );
  }

  const supabase = createSupabaseServerClient();

  const { data, error } = await supabase
    .from("export_configs")
    .insert({
      user_id: user.id,
      provider,
      bucket,
      prefix: prefix ?? "nemulai/",
      region: region ?? null,
      format: format ?? "csv",
      schedule: schedule ?? "weekly",
      access_key_id: provider === "s3" ? await encrypt(access_key_id) : null,
      secret_key: provider === "s3" ? await encrypt(secret_key) : null,
      gcs_credentials: provider === "gcs" ? await encrypt(typeof gcs_credentials === "string" ? gcs_credentials : JSON.stringify(gcs_credentials)) : null,
    })
    .select(
      "id, provider, bucket, prefix, region, format, schedule, is_active, created_at"
    )
    .single();

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  void logAudit({
    userId: user.id,
    action: "export_config.create",
    resourceType: "export_config",
    resourceId: data?.id,
    metadata: { provider, bucket, format: format ?? "csv", schedule: schedule ?? "weekly" },
  });

  return NextResponse.json(data, { status: 201 });
}

// PATCH — update an export config
export async function PATCH(req: NextRequest) {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`settings-export:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const body = await req.json();
  const { id, ...updates } = body;

  if (!id) {
    return NextResponse.json({ error: "id is required" }, { status: 400 });
  }

  const allowed = [
    "bucket",
    "prefix",
    "region",
    "format",
    "schedule",
    "access_key_id",
    "secret_key",
    "gcs_credentials",
    "is_active",
  ];
  const CREDENTIAL_FIELDS = ["access_key_id", "secret_key", "gcs_credentials"];
  const safeUpdates: Record<string, unknown> = {
    updated_at: new Date().toISOString(),
  };
  for (const key of allowed) {
    if (updates[key] !== undefined) {
      if (CREDENTIAL_FIELDS.includes(key) && typeof updates[key] === "string") {
        safeUpdates[key] = await encrypt(updates[key]);
      } else if (key === "gcs_credentials" && typeof updates[key] === "object") {
        safeUpdates[key] = await encrypt(JSON.stringify(updates[key]));
      } else {
        safeUpdates[key] = updates[key];
      }
    }
  }

  const supabase = createSupabaseServerClient();

  const { data, error } = await supabase
    .from("export_configs")
    .update(safeUpdates)
    .eq("id", id)
    .eq("user_id", user.id)
    .select(
      "id, provider, bucket, prefix, region, format, schedule, is_active, updated_at"
    )
    .single();

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  void logAudit({
    userId: user.id,
    action: "export_config.update",
    resourceType: "export_config",
    resourceId: id,
    metadata: { fields_updated: Object.keys(safeUpdates).filter(k => k !== "updated_at") },
  });

  return NextResponse.json(data);
}

// DELETE — delete an export config
export async function DELETE(req: NextRequest) {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`settings-export:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const { id } = await req.json();

  if (!id) {
    return NextResponse.json({ error: "id is required" }, { status: 400 });
  }

  const supabase = createSupabaseServerClient();

  const { error } = await supabase
    .from("export_configs")
    .delete()
    .eq("id", id)
    .eq("user_id", user.id);

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  void logAudit({
    userId: user.id,
    action: "export_config.delete",
    resourceType: "export_config",
    resourceId: id,
  });

  return NextResponse.json({ deleted: true });
}
