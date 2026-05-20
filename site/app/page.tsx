import Link from "next/link";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export default function Home() {
  return (
    <main className="flex min-h-[calc(100vh-57px)] flex-col items-center justify-center gap-8 p-8">
      <div className="text-center space-y-3">
        <h1 className="text-4xl font-bold tracking-tight">torch_activation</h1>
        <p className="text-lg text-muted-foreground max-w-xl">
          300+ activation functions for PyTorch. Searchable, filterable, interactive.
        </p>
      </div>
      <div className="flex gap-3">
        <Link href="/explore" className={cn(buttonVariants({ variant: "default" }))}>
          Explore functions
        </Link>
        <a
          href="https://github.com/hdmquan/torch_activation"
          target="_blank"
          rel="noreferrer"
          className={cn(buttonVariants({ variant: "outline" }))}
        >
          GitHub
        </a>
      </div>
    </main>
  );
}
