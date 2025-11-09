"use client"

import { useReveal } from "@/hooks/use-reveal"

export function ProblemSection() {
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
          <p className="mb-4 font-mono text-sm text-accent md:text-base">/ 01. THE CHALLENGE</p>
          <h2 className="mb-4 font-sans text-5xl font-light leading-tight text-foreground md:text-6xl lg:text-7xl">
            <span className="text-balance">The Plasma Control Problem</span>
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
                Unlimited Clean Energy Awaits
              </h3>
              <p className="leading-relaxed text-foreground/70 md:text-lg">
                Fusion has the potential to deliver <span className="text-foreground/90">virtually limitless, carbon-free electricity</span>, using fuel derived from seawater and producing no long-lived radioactive waste. If achieved, it could end our dependence on fossil fuels and fundamentally reshape the world's energy infrastructure.
              </p>
            </div>

            <div
              className={`transition-all duration-700 ${
                isVisible ? "translate-x-0 opacity-100" : "-translate-x-12 opacity-0"
              }`}
              style={{ transitionDelay: "300ms" }}
            >
              <h3 className="mb-4 font-sans text-2xl font-light text-foreground/90 md:text-3xl">
                Highly Nonlinear Dynamics
              </h3>
              <p className="leading-relaxed text-foreground/70 md:text-lg">
                In a fusion tokamak, the dynamics are <span className="text-foreground/90">highly nonlinear</span> — small changes can trigger massive instabilities. They're <span className="text-foreground/90">coupled across dimensions</span> — magnetic, thermal, and current feedback interact. And they're <span className="text-foreground/90">time-varying</span> — plasma behavior evolves rapidly during the reaction.
              </p>
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
                The Control Challenge
              </h3>
              <p className="leading-relaxed text-foreground/70 md:text-lg">
                Inside a fusion reactor — a tokamak — hydrogen fuel is heated to over <span className="text-foreground/90">100 million degrees Celsius</span>, forming a superheated plasma. This plasma must be <span className="text-foreground/90">kept perfectly suspended by magnetic fields</span>, never touching the reactor walls.
              </p>
              <p className="mt-4 leading-relaxed text-foreground/70 md:text-lg">
                Even tiny instabilities, magnetic drifts, or fluctuations in current can cause the plasma to <span className="text-foreground/90">collapse in milliseconds</span>, halting the reaction and potentially damaging the reactor.
              </p>
            </div>

            <div
              className={`transition-all duration-700 ${
                isVisible ? "translate-x-0 opacity-100" : "-translate-x-12 opacity-0"
              }`}
              style={{ transitionDelay: "400ms" }}
            >
              <h3 className="mb-4 font-sans text-2xl font-light text-foreground/90 md:text-3xl">
                Why PID Controllers Fail
              </h3>
              <p className="leading-relaxed text-foreground/70 md:text-lg">
                A PID controller is <span className="text-foreground/90">reactive</span>, not adaptive. It reacts to errors after they occur, instead of anticipating or learning from them. It assumes the system behaves <span className="text-foreground/90">linearly</span>, which plasma does not. It struggles when control actions are <span className="text-foreground/90">interdependent</span> — like adjusting coils and beams together.
              </p>
              <p className="mt-4 leading-relaxed text-foreground/70 md:text-lg">
                So while PID controllers are still used in fusion experiments for basic subsystems (e.g., coil currents, vacuum pressure), <span className="text-foreground/90">they can't handle global plasma stabilization alone</span>.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
