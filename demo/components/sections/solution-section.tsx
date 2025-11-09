"use client"

import { useReveal } from "@/hooks/use-reveal"

export function SolutionSection() {
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
          <p className="mb-4 font-mono text-sm text-accent md:text-base">/ 02. THE SOLUTION</p>
          <h2 className="mb-4 font-sans text-5xl font-light leading-tight text-foreground md:text-6xl lg:text-7xl">
            <span className="text-balance">Reinforcement Learning for Plasma Control</span>
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
              <h3 className="mb-4 font-sans text-2xl font-light text-foreground/90 md:text-3xl">
                Why Reinforcement Learning?
              </h3>
              <p className="mb-4 leading-relaxed text-foreground/70 md:text-lg">
                Reinforcement Learning, unlike PID, can:
              </p>
              <div className="space-y-3">
                {[
                  "Learn complex, nonlinear dynamics directly from interaction",
                  "Adapt to changes in the system over time",
                  "Optimize multiple objectives simultaneously (e.g., stability and energy efficiency)",
                  "Coordinate multiple actuators intelligently",
                ].map((item, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <div className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-accent" />
                    <p className="text-foreground/70 md:text-lg">{item}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="border-t border-foreground/10 pt-8">
              <div
                className={`transition-all duration-700 ${
                  isVisible ? "translate-y-0 opacity-100" : "translate-y-12 opacity-0"
                }`}
                style={{ transitionDelay: "300ms" }}
              >
                <h3 className="mb-4 font-sans text-2xl font-light text-foreground/90 md:text-3xl">
                  The Bottleneck: Why Online Training Failed
                </h3>
                <p className="mb-4 leading-relaxed text-foreground/70 md:text-lg">
                  <span className="text-foreground/90">TORAX is too slow</span> for online RL training. The physics-based simulation solves coupled PDEs sequentially, making it CPU-bound and impossible to parallelize.
                </p>
                <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-4 font-mono text-sm">
                  <div className="mb-2 text-foreground/90">Online Training Performance:</div>
                  <div className="text-foreground/70">Steps per second: ~2-5 steps/sec</div>
                  <div className="text-foreground/70">Time for 1M steps: ~55-115 hours</div>
                  <div className="text-foreground/70">GPU utilization: &lt;10%</div>
                </div>
                <p className="mt-4 leading-relaxed text-foreground/70 md:text-lg">
                  Training online would take <span className="text-foreground/90">days to weeks</span> per experiment, making hyperparameter tuning impractical and killing research velocity.
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
              <h3 className="mb-4 font-sans text-2xl font-light text-foreground/90 md:text-3xl">
                The Pivot: Offline Reinforcement Learning
              </h3>
              <p className="mb-4 leading-relaxed text-foreground/70 md:text-lg">
                <span className="text-foreground/90">Key Insight: Decouple data collection from training</span>
              </p>
              <div className="space-y-3 leading-relaxed text-foreground/70 md:text-lg">
                <p>
                  <span className="font-semibold text-foreground/90">Step 1:</span> Generate data once using TORAX (slow, but one-time cost)
                </p>
                <p>
                  <span className="font-semibold text-foreground/90">Step 2:</span> Train on that data using GPU-accelerated offline RL (fast, repeatable)
                </p>
              </div>
            </div>

            <div className="border-t border-foreground/10 pt-8">
              <div
                className={`transition-all duration-700 ${
                  isVisible ? "translate-y-0 opacity-100" : "translate-y-12 opacity-0"
                }`}
                style={{ transitionDelay: "400ms" }}
              >
                <h3 className="mb-4 font-sans text-2xl font-light text-foreground/90 md:text-3xl">
                  Conservative Q-Learning (CQL)
                </h3>
                <p className="mb-4 leading-relaxed text-foreground/70 md:text-lg">
                  We use <span className="text-foreground/90">CQL</span> — an offline RL algorithm designed to learn from fixed datasets. It includes a conservative penalty that prevents overestimation of unseen actions, enabling stable training without environment interaction.
                </p>
                <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-4">
                  <div className="mb-2 font-mono text-sm text-foreground/90">Training Performance:</div>
                  <div className="space-y-1 text-sm text-foreground/70">
                    <div><span className="font-semibold">Offline (CQL):</span> ~52 steps/sec, 2.7 hours for 500k steps</div>
                    <div><span className="font-semibold">Speedup:</span> 10-25x faster than online</div>
                    <div><span className="font-semibold">GPU utilization:</span> 60-90% (vs &lt;10% online)</div>
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
