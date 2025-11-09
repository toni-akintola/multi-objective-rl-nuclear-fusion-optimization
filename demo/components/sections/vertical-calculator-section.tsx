"use client"

import { MagneticButton } from "@/components/magnetic-button"
import Link from "next/link"

export function VerticalCalculatorSection() {
  return (
    <section className="flex min-h-screen w-screen shrink-0 snap-start flex-col justify-center px-6 md:px-12 lg:px-16">
      <div className="mx-auto w-full max-w-5xl text-center">
        <h2 className="mb-8 font-sans text-4xl font-light text-foreground md:text-5xl">Impact Calculator</h2>
        <p className="mb-8 text-lg text-foreground/70">
          Calculate the economic impact and importance of vertical displacement event (VDE) prevention systems.
        </p>
        <Link href="/vertical-vis">
          <MagneticButton size="lg" variant="primary">
            Open Calculator
          </MagneticButton>
        </Link>
      </div>
    </section>
  )
}

