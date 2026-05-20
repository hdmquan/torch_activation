import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "torch_activation",
  description: "Explorer for 300+ PyTorch activation functions",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} min-h-screen bg-background text-foreground`}>
        <header className="border-b px-6 py-3 flex items-center justify-between">
          <a href="/" className="font-mono text-sm font-semibold">torch_activation</a>
          <nav className="flex items-center gap-4 text-sm text-muted-foreground">
            <a href="/explore" className="hover:text-foreground">Explore</a>
            <a
              href="https://github.com/hdmquan/torch_activation"
              target="_blank"
              rel="noreferrer"
              className="hover:text-foreground"
            >
              GitHub
            </a>
          </nav>
        </header>
        {children}
      </body>
    </html>
  );
}
