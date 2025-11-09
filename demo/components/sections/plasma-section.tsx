"use client"

import { useReveal } from "@/hooks/use-reveal"

export function PlasmaSection() {
  const { ref: revealRef, isVisible } = useReveal(0.2)

  return (
    <section className="flex min-h-screen w-screen shrink-0 snap-start flex-col px-4 pt-24 pb-24 sm:px-6 sm:pt-32 sm:pb-32 md:px-8 md:pt-32 md:pb-32 lg:px-12 lg:pt-36 lg:pb-36 xl:px-16">
      <div className="mx-auto w-full max-w-7xl">
        {/* Header */}
        <div
          ref={revealRef as React.RefObject<HTMLDivElement>}
          className={`mb-8 transition-all duration-700 md:mb-12 ${
            isVisible ? "translate-y-0 opacity-100" : "translate-y-12 opacity-0"
          }`}
        >
          <p className="mb-4 font-mono text-sm text-accent md:text-base">/ 04. PLASMA MONITORING</p>
          <h2 className="mb-4 font-sans text-5xl font-light leading-tight text-foreground md:text-6xl lg:text-7xl">
            <span className="text-balance">Plasma Shape Monitoring</span>
          </h2>
        </div>

        {/* Content Layout */}
        <div className="grid grid-cols-1 gap-8 md:grid-cols-2 md:gap-12">
          {/* Left Column */}
          <div className="space-y-8">
            <div
              className={`transition-all duration-700 ${
                isVisible ? "translate-x-0 opacity-100" : "-translate-x-12 opacity-0"
              }`}
              style={{ transitionDelay: "100ms" }}
            >
              <h2 className="mb-4 font-sans text-2xl font-light text-foreground/90 md:text-3xl">
                Shape Testing: Real-Time Constraint Monitoring
              </h2>
              <p className="mb-6 leading-relaxed text-foreground/70 md:text-lg">
                Shape testing continuously monitors three plasma parameters that determine stability:
              </p>
              
              <div className="space-y-4">
                <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-4">
                  <div className="mb-2 font-mono text-sm font-semibold text-foreground/90">β_N (Normalized Beta)</div>
                  <p className="text-foreground/70 md:text-lg">
                    Plasma pressure vs magnetic field strength. Too high → disruptions. Safe range: 0.5 - 3.0
                  </p>
                </div>
                
                <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-4">
                  <div className="mb-2 font-mono text-sm font-semibold text-foreground/90">q_min (Minimum Safety Factor)</div>
                  <p className="text-foreground/70 md:text-lg">
                    Prevents internal instabilities. Too low → internal disruptions. Must be ≥ 1.0
                  </p>
                </div>
                
                <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-4">
                  <div className="mb-2 font-mono text-sm font-semibold text-foreground/90">q95 (Edge Safety Factor)</div>
                  <p className="text-foreground/70 md:text-lg">
                    Prevents edge disruptions and ELMs. Safe range: 3.0 - 5.0
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="space-y-8">
            <div
              className={`transition-all duration-700 ${
                isVisible ? "translate-x-0 opacity-100" : "-translate-x-12 opacity-0"
              }`}
              style={{ transitionDelay: "200ms" }}
            >
              <h2 className="mb-4 font-sans text-2xl font-light text-foreground/90 md:text-3xl">
                Why It Matters
              </h2>
              <p className="mb-6 leading-relaxed text-foreground/70 md:text-lg">
                Without monitoring, plasma can drift into unsafe states. The shape guard:
              </p>
              
              <div className="space-y-4">
                <div className="flex gap-3">
                  <div className="flex h-2 w-2 shrink-0 items-center justify-center rounded-full bg-foreground/60 mt-1.5"></div>
                  <div>
                    <div className="mb-1 font-sans text-base font-semibold text-foreground">Detects Violations</div>
                    <p className="text-foreground/70 md:text-lg">
                      Real-time monitoring with 0.1ms overhead per step
                    </p>
                  </div>
                </div>
                
                <div className="flex gap-3">
                  <div className="flex h-2 w-2 shrink-0 items-center justify-center rounded-full bg-foreground/60 mt-1.5"></div>
                  <div>
                    <div className="mb-1 font-sans text-base font-semibold text-foreground">Teaches Self-Correction</div>
                    <p className="text-foreground/70 md:text-lg">
                      Reward shaping teaches the agent what to do. It rewards self-correction, not just avoidance.
                    </p>
                  </div>
                </div>
                
                <div className="flex gap-3">
                  <div className="flex h-2 w-2 shrink-0 items-center justify-center rounded-full bg-foreground/60 mt-1.5"></div>
                  <div>
                    <div className="mb-1 font-sans text-base font-semibold text-foreground">Prevents Disruptions</div>
                    <p className="text-foreground/70 md:text-lg">
                      Reduces disruption rate from 8% to 2%, preventing 60 disruptions per year (75% reduction)
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div
              className={`transition-all duration-700 ${
                isVisible ? "translate-x-0 opacity-100" : "-translate-x-12 opacity-0"
              }`}
              style={{ transitionDelay: "300ms" }}
            >
              <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6">
                <h3 className="mb-4 font-sans text-xl font-semibold text-foreground">Impact: Disruption Prevention</h3>
                <div className="space-y-3 font-mono text-sm">
                  <div className="flex justify-between">
                    <span className="text-foreground/70">Pulses per year:</span>
                    <span className="text-foreground/90">1,000</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-foreground/70">Disruption rate (baseline):</span>
                    <span className="text-foreground/90">8%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-foreground/70">Disruption rate (with guard):</span>
                    <span className="text-foreground/90">2%</span>
                  </div>
                  <div className="border-t border-foreground/10 pt-3 mt-3">
                    <div className="flex justify-between">
                      <span className="text-foreground/70">Disruptions prevented:</span>
                      <span className="text-foreground/90 font-semibold">60/year</span>
                    </div>
                    <div className="flex justify-between mt-2">
                      <span className="text-foreground/70">Cost savings:</span>
                      <span className="text-foreground/90 font-semibold">$6M-30M/year</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

