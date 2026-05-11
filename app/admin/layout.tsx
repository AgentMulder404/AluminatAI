import { redirect } from "next/navigation";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import AdminShell from "./_components/AdminShell";

const ADMIN_EMAILS = (process.env.ADMIN_EMAIL ?? "")
  .split(",")
  .map((e) => e.trim().toLowerCase())
  .filter(Boolean);

export default async function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const supabase = await createSupabaseCookieClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user || !ADMIN_EMAILS.includes(user.email?.toLowerCase() ?? "")) {
    redirect("/");
  }

  return <AdminShell>{children}</AdminShell>;
}
