"use client"

import { useReveal } from "@/hooks/use-reveal"

export function ReferencesSection() {
  const { ref, isVisible } = useReveal(0.2)

  const references = [
    {
      category: "Reinforcement Learning",
      items: [
        {
          title: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor",
          authors: "Haarnoja et al.",
          venue: "ICML 2018",
          link: "https://arxiv.org/abs/1801.01290",
        },
        {
          title: "Conservative Q-Learning for Offline Reinforcement Learning",
          authors: "Kumar et al.",
          venue: "NeurIPS 2020",
          link: "https://arxiv.org/abs/2006.04779",
        },
        {
          title: "Spinning Up in Deep RL",
          authors: "OpenAI",
          venue: "Educational Resource",
          link: "https://spinningup.openai.com/en/latest/algorithms/sac.html",
        },
      ],
    },
    {
      category: "Fusion Physics & Simulation",
      items: [
        {
          title: "TORAX: A Transport Solver for Tokamak Plasmas",
          authors: "TORAX Team",
          venue: "Google Research",
          link: "https://github.com/google/torax",
        },
        {
          title: "gym-TORAX: Reinforcement Learning Environment for Tokamak Control",
          authors: "gym-TORAX Contributors",
          venue: "Open Source",
          link: "https://github.com/google-research/gym-torax",
        },
      ],
    },
    {
      category: "Offline Reinforcement Learning",
      items: [
        {
          title: "Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems",
          authors: "Levine et al.",
          venue: "arXiv 2020",
          link: "https://arxiv.org/abs/2005.01643",
        },
        {
          title: "Offline RL for Real-World Applications",
          authors: "Agarwal et al.",
          venue: "NeurIPS 2020",
          link: "https://arxiv.org/abs/2006.04779",
        },
      ],
    },
    {
      category: "Plasma Control & Tokamak Operations",
      items: [
        {
          title: "Plasma Shape Control in Tokamaks",
          authors: "Various",
          venue: "Fusion Engineering and Design",
          link: "https://www.sciencedirect.com/topics/engineering/plasma-shape-control",
        },
        {
          title: "Real-Time Control of Tokamak Plasmas",
          authors: "Various",
          venue: "Nuclear Fusion",
          link: "https://iopscience.iop.org/journal/0029-5515",
        },
      ],
    },
    {
      category: "Infrastructure & Tools",
      items: [
        {
          title: "Modal: Serverless GPU Infrastructure",
          authors: "Modal Labs",
          venue: "Commercial Platform",
          link: "https://modal.com",
        },
        {
          title: "Stable Baselines3: Reliable Reinforcement Learning Implementations",
          authors: "Raffin et al.",
          venue: "JMLR 2021",
          link: "https://github.com/DLR-RM/stable-baselines3",
        },
        {
          title: "PyTorch: An Imperative Style, High-Performance Deep Learning Library",
          authors: "Paszke et al.",
          venue: "NeurIPS 2019",
          link: "https://pytorch.org",
        },
      ],
    },
  ]

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
          <p className="mb-4 font-mono text-sm text-accent md:text-base">/ REFERENCES</p>
          <h2 className="mb-4 font-sans text-5xl font-light leading-tight text-foreground md:text-6xl lg:text-7xl">
            <span className="text-balance">Citations & Resources</span>
          </h2>
          <p className="max-w-2xl text-foreground/70 md:text-lg">
            Papers, codebases, and resources that this work builds upon.
          </p>
        </div>

        <div className="space-y-12">
          {references.map((category, categoryIdx) => (
            <div
              key={categoryIdx}
              className={`transition-all duration-700 ${
                isVisible ? "translate-y-0 opacity-100" : "translate-y-12 opacity-0"
              }`}
              style={{ transitionDelay: `${100 * (categoryIdx + 1)}ms` }}
            >
              <h3 className="mb-6 font-sans text-2xl font-light text-foreground/90 md:text-3xl">
                {category.category}
              </h3>
              <div className="space-y-4">
                {category.items.map((item, itemIdx) => (
                  <div
                    key={itemIdx}
                    className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 transition-all hover:border-foreground/20 hover:bg-foreground/10"
                  >
                    <a
                      href={item.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block group"
                    >
                      <h4 className="mb-2 font-sans text-lg font-semibold text-foreground group-hover:text-accent transition-colors">
                        {item.title}
                      </h4>
                      <p className="mb-2 text-foreground/70 md:text-base">
                        {item.authors}
                      </p>
                      <p className="text-sm text-foreground/60 font-mono">
                        {item.venue}
                      </p>
                      <p className="mt-2 text-sm text-accent group-hover:underline">
                        {item.link}
                      </p>
                    </a>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

