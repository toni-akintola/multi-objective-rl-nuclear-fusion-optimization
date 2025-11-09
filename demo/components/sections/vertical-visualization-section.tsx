"use client"

import { MagneticButton } from "@/components/magnetic-button"

export function VerticalVisualizationSection() {
  return (
    <section className="flex min-h-screen w-screen shrink-0 snap-start flex-col justify-center px-6 md:px-12 lg:px-16">
      <div className="mx-auto w-full max-w-5xl text-center">
        <h2 className="mb-8 font-sans text-4xl font-light text-foreground md:text-5xl">Live Chamber Visualization</h2>
        <p className="mb-8 text-lg text-foreground/70">
          Watch the tokamak chamber in real-time. See the vertical position (Z) being tracked and controlled as the RL agent prevents VDEs.
        </p>
        <MagneticButton
          size="lg"
          variant="primary"
          onClick={async () => {
            try {
              const response = await fetch("/api/launch-vertical")
              const data = await response.json()
              if (data.success) {
                alert("✅ Rotating 3D vertical guard visualization launched! A matplotlib window should open showing plasma vertical position with interactive coil disable/enable controls.")
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

