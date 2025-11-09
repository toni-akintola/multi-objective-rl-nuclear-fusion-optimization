"use client"

import { useReveal } from "@/hooks/use-reveal"

export function ProblemSection() {
  const { ref, isVisible } = useReveal(0.2)

  return (
    <section
      ref={ref}
      className="flex min-h-screen w-screen shrink-0 snap-start flex-col justify-center px-6 md:px-12 lg:px-16"
    >
      <div className="mx-auto w-full max-w-5xl">
        <div
          className={`mb-12 transition-all duration-700 md:mb-16 ${
            isVisible ? "translate-y-0 opacity-100" : "translate-y-12 opacity-0"
          }`}
        >
          <p className="mb-4 font-mono text-sm text-accent md:text-base">/ 01. THE CHALLENGE</p>
          <h2 className="mb-6 font-sans text-5xl font-light leading-tight text-foreground md:text-6xl lg:text-7xl">
            <span className="text-balance">The Plasma Control Problem</span>
          </h2>
        </div>

        <div className="space-y-8 md:space-y-10">
          <div
            className={`transition-all duration-700 ${
              isVisible ? "translate-x-0 opacity-100" : "-translate-x-12 opacity-0"
            }`}
            style={{ transitionDelay: "100ms" }}
          >
            <h3 className="mb-4 font-sans text-2xl font-light text-foreground/90 md:text-3xl">
              Unlimited Clean Energy Awaits
            </h3>
            <p className="max-w-2xl leading-relaxed text-foreground/70 md:text-lg">
              Fusion has the potential to deliver virtually limitless, carbon-free electricity using fuel derived from
              seawater and producing no long-lived radioactive waste. If achieved, it could end our dependence on fossil
              fuels and fundamentally reshape the world's energy infrastructure.
            </p>
          </div>

          <div
            className={`transition-all duration-700 ${
              isVisible ? "translate-x-0 opacity-100" : "-translate-x-12 opacity-0"
            }`}
            style={{ transitionDelay: "200ms" }}
          >
            <h3 className="mb-4 font-sans text-2xl font-light text-foreground/90 md:text-3xl">
              But One Enormous Challenge Remains
            </h3>
            <p className="max-w-2xl leading-relaxed text-foreground/70 md:text-lg">
              Inside a fusion reactor—a tokamak—hydrogen fuel is heated to over 100 million degrees Celsius, forming a
              superheated plasma. This plasma must be kept perfectly suspended by magnetic fields, never touching the
              reactor walls.
            </p>
          </div>

          <div
            className={`transition-all duration-700 ${
              isVisible ? "translate-x-0 opacity-100" : "-translate-x-12 opacity-0"
            }`}
            style={{ transitionDelay: "300ms" }}
          >
            <h3 className="mb-4 font-sans text-2xl font-light text-foreground/90 md:text-3xl">
              Millisecond Instability Window
            </h3>
            <p className="max-w-2xl leading-relaxed text-foreground/70 md:text-lg">
              Even tiny instabilities, magnetic drifts, or fluctuations in current can cause the plasma to collapse in
              milliseconds, halting the reaction and potentially damaging the reactor.
            </p>
          </div>

          <div className="border-t border-foreground/10 pt-8">
            <div
              className={`transition-all duration-700 ${
                isVisible ? "translate-x-0 opacity-100" : "-translate-x-12 opacity-0"
              }`}
              style={{ transitionDelay: "400ms" }}
            >
              <h4 className="mb-4 font-mono text-sm text-accent">System Complexity</h4>
              <p className="max-w-2xl leading-relaxed text-foreground/70">
                The plasma dynamics are highly <span className="text-foreground/90">nonlinear</span> — small changes can
                trigger massive instabilities. They are{" "}
                <span className="text-foreground/90">coupled across dimensions</span>— magnetic, thermal, and current
                feedback interact. And they are <span className="text-foreground/90">time-varying</span>— plasma
                behavior evolves rapidly during the reaction.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
