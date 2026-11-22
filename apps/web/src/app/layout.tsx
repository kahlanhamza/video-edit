import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DirectorOS Studio",
  description: "Autonomous documentary pipeline — human review surface",
};

// Full-viewport app shell — pages bring their own chrome (top bar / rail /
// docks). No global header, no page scroll: the Studio is an NLE, not a doc.
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="h-screen overflow-hidden antialiased">{children}</body>
    </html>
  );
}
