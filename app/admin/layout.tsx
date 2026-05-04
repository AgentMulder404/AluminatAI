"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/admin", label: "Home" },
  { href: "/admin/marketing", label: "Marketing" },
  { href: "/admin/funding", label: "Funding" },
  { href: "/admin/demo", label: "Demo" },
  { href: "/admin/onboarding", label: "Onboarding" },
  { href: "/admin/tasks", label: "Tasks" },
  { href: "/admin/social", label: "Social" },
  { href: "/admin/outreach", label: "Outreach" },
  { href: "/admin/downloads", label: "Downloads" },
  { href: "/admin/security", label: "Security" },
  { href: "/admin/lawyer", label: "Lawyer" },
  { href: "/admin/financial-advisor", label: "Finance" },
];

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  return (
    <div className="flex min-h-screen bg-neutral-950">
      {/* Sidebar */}
      <nav className="w-52 shrink-0 border-r border-neutral-800 p-4 flex flex-col gap-1">
        <Link
          href="/admin"
          className="text-lg font-bold text-white mb-4 px-3"
        >
          AluminatAI
        </Link>

        {NAV_ITEMS.map((item) => {
          const active =
            item.href === "/admin"
              ? pathname === "/admin"
              : pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                active
                  ? "bg-green-600/20 text-green-400"
                  : "text-neutral-400 hover:text-white hover:bg-neutral-800"
              }`}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Main content */}
      <main className="flex-1 overflow-auto">{children}</main>
    </div>
  );
}
