"use client"

import { MagneticButton } from "@/components/magnetic-button"

export function PlasmaVisualizationSection() {
  return (
    <section className="flex min-h-screen w-screen shrink-0 snap-start flex-col justify-center px-6 md:px-12 lg:px-16">
      <div className="mx-auto w-full max-w-5xl text-center">
        <h2 className="mb-8 font-sans text-4xl font-light text-foreground md:text-5xl">Live Chamber Visualization</h2>
        <p className="mb-8 text-lg text-foreground/70">
          Watch the tokamak chamber in real-time. See particle trails, plasma boundary, and shape parameters as the RL agent controls the plasma.
        </p>
        <MagneticButton
          size="lg"
          variant="primary"
          onClick={async () => {
            try {
              const response = await fetch("/api/launch-python")
              const data = await response.json()
              if (data.success) {
                alert("Visualization launched! A matplotlib window should open showing the live chamber simulation.")
              } else {
                alert(`Error: ${data.error}`)
              }
            } catch (error) {
              alert("Failed to launch visualization. Make sure Python and dependencies are installed.")
            }
          }}
        >
          Launch Visualization
        </MagneticButton>
      </div>
    </section>
  )
}

