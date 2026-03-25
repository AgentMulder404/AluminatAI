import Link from "next/link";

export default function Footer() {
  return (
    <footer className="border-t border-neutral-800 bg-neutral-950 py-6 px-6">
      <div className="max-w-5xl mx-auto flex flex-wrap gap-x-6 gap-y-2 text-xs text-neutral-500">
        <Link href="/" className="hover:text-neutral-300">
          AluminatiAI
        </Link>
        <Link href="/benchmarks" className="hover:text-neutral-300">
          Benchmarks
        </Link>
        <Link href="/dashboard" className="hover:text-neutral-300">
          Dashboard
        </Link>
      </div>
    </footer>
  );
}
