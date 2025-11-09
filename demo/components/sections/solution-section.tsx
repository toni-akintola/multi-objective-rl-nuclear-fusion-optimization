"use client"

import { useReveal } from "@/hooks/use-reveal"

export function SolutionSection() {
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
          <p className="mb-4 font-mono text-sm text-accent md:text-base">/ 02. THE APPROACH</p>
          <h2 className="mb-4 font-sans text-5xl font-light leading-tight text-foreground md:text-6xl lg:text-7xl">
            <span className="text-balance">Why Traditional Control Fails</span>
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
                PID Controllers Are Reactive, Not Adaptive
              </h3>
              <div className="space-y-3 leading-relaxed text-foreground/70 md:text-lg">
                <p>
                  A PID controller is <span className="text-foreground/90">reactive</span>, not adaptive. It reacts to
                  errors after they occur, instead of anticipating or learning from them.
                </p>
                <p>
                  It assumes the system behaves <span className="text-foreground/90">linearly</span>, which plasma does
                  not. It struggles when control actions are <span className="text-foreground/90">interdependent</span> —
                  like adjusting coils and beams together.
                </p>
                <p className="pt-2">
                  So while PID controllers are still used in fusion experiments for basic subsystems, they can't handle
                  global plasma stabilization alone.
                </p>
              </div>
            </div>

            <div className="border-t border-foreground/10 pt-8">
              <div
                className={`transition-all duration-700 ${
                  isVisible ? "translate-y-0 opacity-100" : "translate-y-12 opacity-0"
                }`}
                style={{ transitionDelay: "500ms" }}
              >
                <h3 className="mb-6 font-sans text-2xl font-light text-foreground/90 md:text-3xl">
                  Soft Actor-Critic: Building Adaptive Controllers
                </h3>
                <p className="mb-4 leading-relaxed text-foreground/70 md:text-lg">
                  Our controller is based on Soft Actor-Critic (SAC)—a modern RL algorithm built for continuous control. The
                  actor learns by maximizing not just the reward from the environment, but also{" "}
                  <span className="text-foreground/90">entropy</span>—a measure of uncertainty or randomness in its
                  decisions.
                </p>
                <p className="leading-relaxed text-foreground/70 md:text-lg">
                  In most RL algorithms, the goal is to find a deterministic policy: always take the single best action for
                  each state. But in complex, chaotic systems like plasma physics, being too deterministic too early can
                  trap the agent in suboptimal behavior. SAC keeps the exploration window open, discovering better
                  strategies over time.
                </p>
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="space-y-8">
            <div className="border-t border-foreground/10 pt-8">
              <div
                className={`transition-all duration-700 ${
                  isVisible ? "translate-y-0 opacity-100" : "translate-y-12 opacity-0"
                }`}
                style={{ transitionDelay: "200ms" }}
              >
                <p className="mb-8 font-mono text-sm text-accent md:text-base">Reinforcement Learning Can:</p>
                <div className="space-y-4">
                  {[
                    "Learn complex, nonlinear dynamics directly from interaction",
                    "Adapt to changes in the system over time",
                    "Optimize multiple objectives simultaneously (e.g., stability and energy efficiency)",
                    "Coordinate multiple actuators intelligently",
                  ].map((item, i) => (
                    <div
                      key={i}
                      className={`flex items-start gap-4 transition-all duration-700 ${
                        isVisible ? "translate-x-0 opacity-100" : "-translate-x-8 opacity-0"
                      }`}
                      style={{ transitionDelay: `${300 + i * 100}ms` }}
                    >
                      <div className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-accent" />
                      <p className="text-foreground/70 md:text-lg">{item}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
