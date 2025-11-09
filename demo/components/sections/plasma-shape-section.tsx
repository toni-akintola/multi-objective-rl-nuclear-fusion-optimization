"use client"

export function PlasmaShapeSection() {
  return (
    <section className="flex min-h-screen w-screen shrink-0 snap-start flex-col justify-center px-6 md:px-12 lg:px-16">
      <div className="mx-auto w-full max-w-5xl">
        <h2 className="mb-8 font-sans text-4xl font-light text-foreground md:text-5xl">Plasma Shape Parameters</h2>
        <div className="grid gap-6 md:grid-cols-3">
          <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-md">
            <div className="mb-2 font-mono text-sm font-semibold text-blue-400">β_N</div>
            <div className="mb-2 font-sans text-lg font-medium text-foreground">Normalized Beta</div>
            <p className="text-sm text-foreground/70">
              Plasma pressure vs magnetic field strength. Safe range: 0.5 - 3.0
            </p>
          </div>
          <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-md">
            <div className="mb-2 font-mono text-sm font-semibold text-green-400">q_min</div>
            <div className="mb-2 font-sans text-lg font-medium text-foreground">Minimum Safety Factor</div>
            <p className="text-sm text-foreground/70">
              Prevents internal instabilities. Must be ≥ 1.0
            </p>
          </div>
          <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-md">
            <div className="mb-2 font-mono text-sm font-semibold text-orange-400">q95</div>
            <div className="mb-2 font-sans text-lg font-medium text-foreground">Edge Safety Factor</div>
            <p className="text-sm text-foreground/70">
              Prevents edge disruptions and ELMs. Safe range: 3.0 - 5.0
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

