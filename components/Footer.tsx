import Link from "next/link";

export default function Footer() {
  return (
    <footer className="border-t border-neutral-800 bg-neutral-950 px-6 py-12">
      <div className="max-w-6xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8 text-sm">
        <div>
          <div className="font-semibold text-neutral-300 mb-3">Product</div>
          <div className="space-y-2 text-neutral-500">
            <div><Link href="/dashboard" className="hover:text-white transition-colors">Dashboard</Link></div>
            <div><Link href="/benchmarks" className="hover:text-white transition-colors">Benchmarks</Link></div>
            <div><Link href="/carbon" className="hover:text-white transition-colors">Carbon</Link></div>
            <div><Link href="/pricing" className="hover:text-white transition-colors">Pricing</Link></div>
          </div>
        </div>
        <div>
          <div className="font-semibold text-neutral-300 mb-3">Company</div>
          <div className="space-y-2 text-neutral-500">
            <div><Link href="/enterprise" className="hover:text-white transition-colors">Enterprise</Link></div>
            <div><Link href="/legal/security-questionnaire" className="hover:text-white transition-colors">Security</Link></div>
            <div><Link href="/legal/msa" className="hover:text-white transition-colors">Legal</Link></div>
          </div>
        </div>
        <div>
          <div className="font-semibold text-neutral-300 mb-3">Open Source</div>
          <div className="space-y-2 text-neutral-500">
            <div><a href="https://github.com/AgentMulder404/NemulAI" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">GitHub</a></div>
            <div><a href="https://github.com/AgentMulder404/NemulAI#quick-start" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">Install Guide</a></div>
            <div><a href="https://github.com/AgentMulder404/NemulAI#api" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">API Docs</a></div>
          </div>
        </div>
        <div>
          <div className="font-semibold text-neutral-300 mb-3">Connect</div>
          <div className="space-y-2 text-neutral-500">
            <div><a href="https://x.com/NemulAI_Dev" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">X / Twitter</a></div>
            <div><a href="https://linkedin.com/company/nemulai" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">LinkedIn</a></div>
          </div>
        </div>
      </div>
      <div className="max-w-6xl mx-auto mt-8 pt-8 border-t border-neutral-800 text-sm text-neutral-600 text-center">
        2026 NemulAI
      </div>
    </footer>
  );
}
