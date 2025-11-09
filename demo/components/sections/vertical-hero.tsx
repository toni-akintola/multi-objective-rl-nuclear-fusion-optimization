"use client"

import { MagneticButton } from "@/components/magnetic-button"
import Link from "next/link"

export function VerticalHeroSection() {
  return (
    <section className="flex min-h-screen w-screen shrink-0 flex-col justify-end px-6 pb-16 pt-24 md:px-12 md:pb-24">
      <div className="max-w-3xl">
        <div className="mb-4 inline-block animate-in fade-in slide-in-from-bottom-4 rounded-full border border-foreground/20 bg-foreground/15 px-4 py-1.5 backdrop-blur-md duration-700">
          <p className="font-mono text-xs text-foreground/90">Vertical Displacement Control</p>
        </div>
        <h1 className="mb-6 animate-in fade-in slide-in-from-bottom-8 font-sans text-6xl font-light leading-[1.1] tracking-tight text-foreground duration-1000 md:text-7xl lg:text-8xl">
          <span className="text-balance">
            Preventing
            <br />
            vertical disruptions
          </span>
        </h1>
        <p className="mb-8 max-w-xl animate-in fade-in slide-in-from-bottom-4 text-lg leading-relaxed text-foreground/80 duration-1000 delay-200 md:text-xl">
          <span className="text-pretty">
            Advanced control systems to detect and prevent Vertical Displacement Events (VDEs) before they cause catastrophic damage to the reactor.
          </span>
        </p>
        <div className="flex animate-in fade-in slide-in-from-bottom-4 flex-col gap-4 duration-1000 delay-300 sm:flex-row sm:items-center">
          <MagneticButton size="lg" variant="primary">
            Explore VDEs
          </MagneticButton>
          <MagneticButton size="lg" variant="secondary">
            View Simulation
          </MagneticButton>
        </div>
      </div>

      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-in fade-in duration-1000 delay-500">
        <div className="flex items-center gap-2">
          <p className="font-mono text-xs text-foreground/80">Scroll to explore</p>
          <div className="flex h-6 w-12 items-center justify-center rounded-full border border-foreground/20 bg-foreground/15 backdrop-blur-md">
            <div className="h-2 w-2 animate-pulse rounded-full bg-foreground/80" />
          </div>
        </div>
      </div>
    </section>
  )
}

