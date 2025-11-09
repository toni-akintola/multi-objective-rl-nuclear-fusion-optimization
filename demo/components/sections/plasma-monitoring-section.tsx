"use client"

export function PlasmaMonitoringSection() {
  return (
    <section className="flex min-h-screen w-screen shrink-0 snap-start flex-col justify-center px-6 md:px-12 lg:px-16">
      <div className="mx-auto w-full max-w-5xl">
        <h2 className="mb-8 font-sans text-4xl font-light text-foreground md:text-5xl">Why Monitoring Matters</h2>
        <div className="space-y-6">
          <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-md">
            <h3 className="mb-2 font-sans text-xl font-semibold text-foreground">Prevents Disruptions</h3>
            <p className="text-foreground/70">
              Shape violations can lead to sudden loss of plasma confinement, causing millions in damage and weeks of downtime.
            </p>
          </div>
          <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-md">
            <h3 className="mb-2 font-sans text-xl font-semibold text-foreground">Enables Self-Correction</h3>
            <p className="text-foreground/70">
              The RL agent learns to automatically adjust coil currents when parameters drift, bringing plasma back to safe states.
            </p>
          </div>
          <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-md">
            <h3 className="mb-2 font-sans text-xl font-semibold text-foreground">Massive Cost Savings</h3>
            <p className="text-foreground/70">
              Prevents 60-80 disruptions per year, saving $60M+ annually in real tokamak operations.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

