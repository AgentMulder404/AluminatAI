// GET /api/notifications — list user's notifications
// PATCH /api/notifications — mark notifications as read

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";

export const runtime = "edge";

export async function GET(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const params = req.nextUrl.searchParams;
  const unreadOnly = params.get("unread") === "true";
  const limit = Math.min(Number(params.get("limit") ?? 20), 100);
  const offset = Number(params.get("offset") ?? 0);

  const supabase = createSupabaseServerClient();

  let query = supabase
    .from("notifications")
    .select("id, type, title, message, read, metadata, created_at", {
      count: "exact",
    })
    .eq("user_id", user.id)
    .order("created_at", { ascending: false })
    .range(offset, offset + limit - 1);

  if (unreadOnly) {
    query = query.eq("read", false);
  }

  const { data: notifications, count, error } = await query;

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  // Also get unread count
  const { count: unreadCount } = await supabase
    .from("notifications")
    .select("id", { count: "exact", head: true })
    .eq("user_id", user.id)
    .eq("read", false);

  return NextResponse.json({
    notifications: notifications ?? [],
    unread_count: unreadCount ?? 0,
    total: count ?? 0,
  });
}

export async function PATCH(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  let body: { ids?: string[]; all?: boolean };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const supabase = createSupabaseServerClient();

  if (body.all) {
    const { error, count } = await supabase
      .from("notifications")
      .update({ read: true })
      .eq("user_id", user.id)
      .eq("read", false);

    if (error) {
      return NextResponse.json({ error: error.message }, { status: 500 });
    }
    return NextResponse.json({ updated: count ?? 0 });
  }

  if (body.ids && Array.isArray(body.ids) && body.ids.length > 0) {
    const { error, count } = await supabase
      .from("notifications")
      .update({ read: true })
      .eq("user_id", user.id)
      .in("id", body.ids);

    if (error) {
      return NextResponse.json({ error: error.message }, { status: 500 });
    }
    return NextResponse.json({ updated: count ?? 0 });
  }

  return NextResponse.json({ error: "Provide ids array or all: true" }, { status: 400 });
}
