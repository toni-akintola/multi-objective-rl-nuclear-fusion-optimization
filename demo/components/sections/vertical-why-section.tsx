"use client"

export function VerticalWhySection() {
  return (
    <section className="flex min-h-screen w-screen shrink-0 snap-start flex-col justify-center px-6 md:px-12 lg:px-16">
      <div className="mx-auto w-full max-w-5xl">
        <h2 className="mb-8 font-sans text-4xl font-light text-foreground md:text-5xl">Why Vertical Monitoring is Critical</h2>
        
        <div className="space-y-6">
          <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-md">
            <h3 className="mb-2 font-sans text-xl font-semibold text-foreground">Prevents VDEs (Vertical Displacement Events)</h3>
            <p className="text-foreground/70">
              When plasma moves too far up or down, it can hit the chamber walls in milliseconds. This causes instant plasma loss and severe damage costing $5M+ and 3-6 months of downtime.
            </p>
          </div>
          
          <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-md">
            <h3 className="mb-2 font-sans text-xl font-semibold text-foreground">Real-Time Correction</h3>
            <p className="text-foreground/70">
              The RL agent learns to detect early signs of vertical instability and automatically adjusts the vertical field coils to bring the plasma back to center before a VDE occurs.
            </p>
          </div>
          
          <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-md">
            <h3 className="mb-2 font-sans text-xl font-semibold text-foreground">Massive Impact</h3>
            <p className="text-foreground/70">
              With vertical guard enabled, VDE rates drop by 95%, episodes run 3x longer, and tokamaks save $50M+ annually in avoided repairs and downtime.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

