"use client"

import { useReveal } from "@/hooks/use-reveal"

export function ApproachSection() {
  const { ref, isVisible } = useReveal(0.2)

  return (
    <section
      ref={ref}
      className="flex min-h-screen w-screen shrink-0 snap-start flex-col px-6 pt-24 pb-24 md:px-12 md:pt-32 md:pb-32 lg:px-16 lg:pt-36 lg:pb-36"
    >
      <div className="mx-auto w-full max-w-5xl">
        <div
          className={`mb-8 transition-all duration-700 md:mb-12 ${
            isVisible ? "translate-y-0 opacity-100" : "translate-y-12 opacity-0"
          }`}
        >
          <p className="mb-4 font-mono text-sm text-accent md:text-base">/ 03. OUR APPROACH</p>
          <h2 className="mb-4 font-sans text-5xl font-light leading-tight text-foreground md:text-6xl lg:text-7xl">
            <span className="text-balance">Offline RL Training Pipeline</span>
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
              <h3 className="mb-4 font-mono text-sm text-accent">Step 1: Data Generation with TORAX</h3>
              <p className="leading-relaxed text-foreground/70 md:text-lg">
                Generate simulation data once using <span className="text-foreground/90">gym-TORAX</span> — a physics-accurate tokamak environment. We collected <span className="text-foreground/90">3.86M transitions</span> across 10 episodes, including observations (plasma parameters), actions (Ip, NBI, ECRH controls), and rewards.
              </p>
              <p className="mt-3 leading-relaxed text-foreground/70 md:text-lg">
                Data collection took <span className="text-foreground/90">~30 minutes</span> (one-time cost), but this data can be reused for unlimited training iterations.
              </p>
            </div>

            <div
              className={`transition-all duration-700 ${
                isVisible ? "translate-x-0 opacity-100" : "-translate-x-12 opacity-0"
              }`}
              style={{ transitionDelay: "300ms" }}
            >
              <h3 className="mb-4 font-mono text-sm text-accent">Step 3: GPU-Scale Training on Modal</h3>
              <p className="leading-relaxed text-foreground/70 md:text-lg">
                Train CQL on <span className="text-foreground/90">4x A100 GPUs</span> with optimized settings: batch size 4096, TF32 precision, and observation/action/reward scaling. Training completes in <span className="text-foreground/90">2.7 hours</span> for 500k steps with 60-90% GPU utilization.
              </p>
            </div>

            <div className="border-t border-foreground/10 pt-8">
              <div
                className={`transition-all duration-700 ${
                  isVisible ? "translate-y-0 opacity-100" : "translate-y-12 opacity-0"
                }`}
                style={{ transitionDelay: "500ms" }}
              >
                <h3 className="mb-4 font-mono text-sm text-accent">Step 5: Online Evaluation</h3>
                <p className="leading-relaxed text-foreground/70 md:text-lg">
                  After training, we evaluate the CQL policy in the <span className="text-foreground/90">actual online gym-TORAX environment</span>. The trained policy generalizes to the live simulation without fine-tuning, demonstrating that offline RL successfully learned the plasma control dynamics.
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
              <h3 className="mb-4 font-mono text-sm text-accent">Step 2: Offline Training with CQL</h3>
              <p className="leading-relaxed text-foreground/70 md:text-lg">
                Train <span className="text-foreground/90">Conservative Q-Learning (CQL)</span> on the fixed dataset. CQL includes a conservative penalty that prevents overestimation of unseen actions, enabling stable training without environment interaction.
              </p>
              <div className="mt-4 rounded-lg border border-foreground/10 bg-foreground/5 p-4 font-mono text-sm">
                <div className="text-foreground/90 mb-2">Key Optimizations:</div>
                <div className="space-y-1 text-foreground/70 text-xs">
                  <div>• StandardObservationScaler</div>
                  <div>• MinMaxActionScaler</div>
                  <div>• StandardRewardScaler</div>
                  <div>• TF32 precision (3x speedup)</div>
                  <div>• Large batch sizes (4096)</div>
                </div>
              </div>
            </div>

            <div
              className={`transition-all duration-700 ${
                isVisible ? "translate-x-0 opacity-100" : "-translate-x-12 opacity-0"
              }`}
              style={{ transitionDelay: "400ms" }}
            >
              <h3 className="mb-4 font-mono text-sm text-accent">Step 4: Reward Signal</h3>
              <p className="leading-relaxed text-foreground/70 md:text-lg">
                Reward measures how closely the plasma stays to its target profile and how stable it remains. The agent optimizes for <span className="text-foreground/90">fusion gain, H98, q_min, and q95</span> — balancing stability and energy efficiency.
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
                  The result: a controller that learns the deep structure of plasma dynamics and adapts in real-time to new conditions, bringing us closer to practical, sustainable fusion energy.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
