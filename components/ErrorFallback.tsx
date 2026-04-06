"use client";

interface ErrorFallbackProps {
  error: Error & { digest?: string };
  reset: () => void;
  title?: string;
}

export default function ErrorFallback({
  error,
  reset,
  title = "Something went wrong",
}: ErrorFallbackProps) {
  return (
    <div className="flex flex-col items-center justify-center py-20 px-6 text-center">
      <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-8 max-w-md">
        <h2 className="text-lg font-semibold text-neutral-100 mb-2">{title}</h2>
        <p className="text-sm text-neutral-400 mb-4">
          {error.message || "An unexpected error occurred."}
        </p>
        {error.digest && (
          <p className="text-xs text-neutral-600 mb-4 font-mono">
            Error ID: {error.digest}
          </p>
        )}
        <div className="flex items-center justify-center gap-3">
          <button
            onClick={reset}
            className="px-4 py-2 text-sm bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition"
          >
            Try Again
          </button>
          <a
            href="/dashboard"
            className="px-4 py-2 text-sm bg-neutral-800 hover:bg-neutral-700 text-neutral-200 rounded-lg transition"
          >
            Go to Dashboard
          </a>
        </div>
      </div>
    </div>
  );
}
