"use client"

import { useReveal } from "@/hooks/use-reveal"

export function ApproachSection() {
  const { ref, isVisible } = useReveal(0.2)

  return (
    <section
      ref={ref}
      className="flex min-h-screen w-screen shrink-0 snap-start flex-col justify-center px-6 md:px-12 lg:px-16"
    >
      <div className="mx-auto w-full max-w-5xl">
        <div
          className={`mb-8 transition-all duration-700 md:mb-12 ${
            isVisible ? "translate-y-0 opacity-100" : "translate-y-12 opacity-0"
          }`}
        >
          <p className="mb-4 font-mono text-sm text-accent md:text-base">/ 03. OUR IMPLEMENTATION</p>
          <h2 className="mb-4 font-sans text-5xl font-light leading-tight text-foreground md:text-6xl lg:text-7xl">
            <span className="text-balance">Training in the Simulation</span>
          </h2>
        </div>

        <div className="grid grid-cols-1 gap-8 md:grid-cols-2 md:gap-12">
          {/* Left Column */}
          <div className="space-y-8">
            <div
              className={`transition-all duration-700 ${
                isVisible ? "translate-x-0 opacity-100" : "-translate-x-12 opacity-0"
              }`}
              style={{ transitionDelay: "100ms" }}
            >
              <h3 className="mb-4 font-mono text-sm text-accent">Step 1: Environment Setup</h3>
              <p className="leading-relaxed text-foreground/70 md:text-lg">
                We train an RL agent (Soft Actor-Critic) inside the <span className="text-foreground/90">gym-TORAX</span>{" "}
                simulation—a physics-accurate tokamak environment built from real fusion data.
              </p>
            </div>

            <div
              className={`transition-all duration-700 ${
                isVisible ? "translate-x-0 opacity-100" : "-translate-x-12 opacity-0"
              }`}
              style={{ transitionDelay: "300ms" }}
            >
              <h3 className="mb-4 font-mono text-sm text-accent">Step 3: Control Actions</h3>
              <p className="leading-relaxed text-foreground/70 md:text-lg">
                The agent learns to output control actions: adjusting coil currents, ECRH (electron cyclotron resonance
                heating) power, and NBI (neutral beam injection) power to maintain stable, efficient plasma.
              </p>
            </div>

            <div className="border-t border-foreground/10 pt-8">
              <div
                className={`transition-all duration-700 ${
                  isVisible ? "translate-y-0 opacity-100" : "translate-y-12 opacity-0"
                }`}
                style={{ transitionDelay: "500ms" }}
              >
                <h3 className="mb-4 font-mono text-sm text-accent">Step 5: Distributed Training at Scale</h3>
                <p className="mb-4 leading-relaxed text-foreground/70 md:text-lg">
                  Training the SAC agent efficiently required substantial computational resources. We leveraged{" "}
                  <span className="text-foreground/90">Modal</span>'s multi-container orchestration to parallelize
                  training across distributed GPU infrastructure, managing complex workloads and data pipelines
                  automatically.
                </p>
                <p className="leading-relaxed text-foreground/70 md:text-lg">
                  This approach allowed us to iterate quickly and scale experiments, resulting in an agent that performs{" "}
                  <span className="text-foreground/90">8x better than random search</span>—a substantial improvement in
                  control performance over naive baseline strategies.
                </p>
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
              <h3 className="mb-4 font-mono text-sm text-accent">Step 2: Physical Signals</h3>
              <p className="leading-relaxed text-foreground/70 md:text-lg">
                The environment provides rich physical signals: plasma current, electron temperature, magnetic field
                profiles, safety factor, and more. The agent observes and learns from these signals.
              </p>
            </div>

            <div
              className={`transition-all duration-700 ${
                isVisible ? "translate-x-0 opacity-100" : "-translate-x-12 opacity-0"
              }`}
              style={{ transitionDelay: "400ms" }}
            >
              <h3 className="mb-4 font-mono text-sm text-accent">Step 4: Reward Signal</h3>
              <p className="leading-relaxed text-foreground/70 md:text-lg">
                Reward is designed to measure how closely the plasma stays to its target profile and how stable it
                remains. The agent is incentivized to minimize crashes while maintaining efficiency.
              </p>
            </div>

            <div className="border-t border-foreground/10 pt-8">
              <div
                className={`transition-all duration-700 ${
                  isVisible ? "translate-y-0 opacity-100" : "translate-y-12 opacity-0"
                }`}
                style={{ transitionDelay: "600ms" }}
              >
                <p className="leading-relaxed text-foreground/70 md:text-lg">
                  Through iterative training, the agent develops a control policy that can anticipate plasma instabilities
                  and correct them before they occur—something traditional PID controllers simply cannot do.
                  Our result is a controller that learns the deep structure of plasma dynamics and adapts in real-time to
                  new conditions, bringing us closer to the dream of practical, sustainable fusion energy.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
