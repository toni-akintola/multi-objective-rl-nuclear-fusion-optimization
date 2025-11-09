"use client"

export function VerticalWhatSection() {
  return (
    <section className="flex min-h-screen w-screen shrink-0 snap-start flex-col justify-center px-6 md:px-12 lg:px-16">
      <div className="mx-auto w-full max-w-5xl">
        <h2 className="mb-8 font-sans text-4xl font-light text-foreground md:text-5xl">What is Vertical Position?</h2>
        <div className="space-y-6">
          <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-md">
            <div className="mb-2 font-mono text-sm font-semibold text-blue-400">Z Position</div>
            <div className="mb-2 font-sans text-lg font-medium text-foreground">Vertical Location</div>
            <p className="text-sm text-foreground/70">
              Height of plasma center in the tokamak. Safe range: -0.1m to +0.1m from magnetic axis
            </p>
          </div>
          <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-md">
            <div className="mb-2 font-mono text-sm font-semibold text-green-400">dZ/dt</div>
            <div className="mb-2 font-sans text-lg font-medium text-foreground">Vertical Velocity</div>
            <p className="text-sm text-foreground/70">
              How fast the plasma is moving up or down. Must be kept below 2.0 m/s
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

