"use client";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="en">
      <body className="bg-neutral-950 text-neutral-100 antialiased">
        <div className="flex flex-col items-center justify-center min-h-screen py-20 px-6 text-center">
          <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-8 max-w-md">
            <h2 className="text-lg font-semibold text-neutral-100 mb-2">
              Something went wrong
            </h2>
            <p className="text-sm text-neutral-400 mb-4">
              {error.message || "A critical error occurred."}
            </p>
            <button
              onClick={reset}
              className="px-4 py-2 text-sm bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition"
            >
              Try Again
            </button>
          </div>
        </div>
      </body>
    </html>
  );
}
