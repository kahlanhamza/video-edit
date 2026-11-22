import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DirectorOS Studio",
  description: "Autonomous documentary pipeline — human review surface",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <header className="border-b border-slate-800 px-6 py-3">
          <a href="/" className="font-bold tracking-wide text-amber-400">
            DirectorOS <span className="text-slate-300">Studio</span>
          </a>
          <span className="ml-3 text-xs text-slate-500">
            M3 — human review gate over the autonomous pipeline
          </span>
          <a
            href="/settings"
            className="btn-ghost float-right !py-1 text-xs"
            title="Provider credentials + connection tests"
          >
            Settings
          </a>
        </header>
        <main className="mx-auto max-w-5xl p-6">{children}</main>
      </body>
    </html>
  );
}
