"use client"

import { useMemo, useRef, useEffect, useState } from "react"
import { notFound } from "next/navigation"
import { Shader, Swirl, ChromaFlow } from "shaders/react"
import { CustomCursor } from "@/components/custom-cursor"
import { GrainOverlay } from "@/components/grain-overlay"
import { MagneticButton } from "@/components/magnetic-button"
import Link from "next/link"
import RandomAgentLineChart, { PIDAgentLineChart } from "@/components/visualizations/random-agent-line"

const AGENT_CONTENT = {
  sac: {
    title: "SAC Agent Visualizations",
    subtitle: "The SAC (Soft Actor-Critic) agent maximizes tradeoff between reward and entropy."
  },
  pid: {
    title: "PID Agent Visualizations",
    subtitle: "The PID (Proportional-Integral-Derivative) agent linearly minimizes error between the target and actual current density.",
  },
  random: {
    title: "Random Agent Visualizations",
    subtitle: "The Random agent takes random actions in environment. Serves as a baseline for comparison.",
  },
} as const

type AgentKey = keyof typeof AGENT_CONTENT

interface AgentPageProps {
  params: {
    agent: string
  }
}

export default function AgentVisualizationsPage({ params }: AgentPageProps) {
  const agentKey = params.agent as AgentKey
  const agentContent = AGENT_CONTENT[agentKey]

  if (!agentContent) {
    notFound()
  }

  const shaderContainerRef = useRef<HTMLDivElement>(null)
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    const checkShaderReady = () => {
      if (shaderContainerRef.current) {
        const canvas = shaderContainerRef.current.querySelector("canvas")
        if (canvas && canvas.width > 0 && canvas.height > 0) {
          setIsLoaded(true)
          return true
        }
      }
      return false
    }

    if (checkShaderReady()) return

    const intervalId = setInterval(() => {
      if (checkShaderReady()) {
        clearInterval(intervalId)
      }
    }, 100)

    const fallbackTimer = setTimeout(() => {
      setIsLoaded(true)
    }, 1500)

    return () => {
      clearInterval(intervalId)
      clearTimeout(fallbackTimer)
    }
  }, [])

  const peerLinks = useMemo(() => Object.entries(AGENT_CONTENT) as Array<[AgentKey, (typeof AGENT_CONTENT)[AgentKey]]>, [])

  return (
    <main className="relative min-h-screen w-full overflow-hidden bg-background text-foreground">
      <CustomCursor />
      <GrainOverlay />

      <div
        ref={shaderContainerRef}
        className={`fixed inset-0 z-0 transition-opacity duration-700 ${isLoaded ? "opacity-100" : "opacity-0"}`}
        style={{ contain: "strict" }}
      >
        <Shader className="h-full w-full">
          <Swirl
            colorA="#1275d8"
            colorB="#e19136"
            speed={0.8}
            detail={0.8}
            blend={50}
            coarseX={40}
            coarseY={40}
            mediumX={40}
            mediumY={40}
            fineX={40}
            fineY={40}
          />
          <ChromaFlow
            baseColor="#0066ff"
            upColor="#0066ff"
            downColor="#d1d1d1"
            leftColor="#e19136"
            rightColor="#e19136"
            intensity={0.9}
            radius={1.8}
            momentum={25}
            maskType="alpha"
            opacity={0.97}
          />
        </Shader>
        <div className="absolute inset-0 bg-black/20" />
      </div>

      <nav
        className={`fixed inset-x-0 top-0 z-50 flex items-center justify-between px-6 py-6 transition-opacity duration-700 md:px-12 ${
          isLoaded ? "opacity-100" : "opacity-0"
        }`}
      >
        <Link href="/" className="group flex items-center gap-2 transition-transform hover:scale-105">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/15 backdrop-blur-md transition-all duration-300 group-hover:scale-110 group-hover:bg-foreground/25">
            <span className="font-sans text-xl font-bold text-foreground">ϕ</span>
          </div>
          <span className="font-sans text-xl font-semibold tracking-tight text-foreground">Fusion Lab</span>
        </Link>
        <div className="hidden items-center gap-6 md:flex">
          {peerLinks.map(([key, config]) => (
            <Link
              key={key}
              href={`/visualizations/${key}`}
              className={`font-sans text-sm font-medium transition-colors ${
                key === agentKey ? "text-foreground" : "text-foreground/80 hover:text-foreground"
              }`}
            >
              {config.title.replace(" Visualizations", "")}
            </Link>
          ))}
        </div>
        <MagneticButton size="default" variant="secondary" onClick={() => window.history.back()}>
          Back
        </MagneticButton>
      </nav>

      <div
        className={`relative z-10 mt-32 flex flex-col gap-12 px-6 pb-24 transition-opacity duration-700 md:px-12 lg:px-24 ${
          isLoaded ? "opacity-100" : "opacity-0"
        }`}
      >
        <section className="flex min-h-[50vh] flex-col justify-end gap-6 pt-12">
          <div className="max-w-3xl space-y-6">
            <h1 className="font-sans text-6xl font-light leading-[1.05] tracking-tight md:text-7xl lg:text-8xl">
              {agentContent.title}
            </h1>
            <p className="max-w-2xl text-lg leading-relaxed text-foreground/80 md:text-xl">{agentContent.subtitle}</p>
          </div>
        </section>



        <section className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
          <div className="rounded-3xl border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-xl md:p-8">
            <div className="space-y-4">
              <div className="inline-flex items-center gap-2 rounded-full border border-foreground/15 bg-foreground/10 px-3 py-1 text-xs uppercase tracking-widest text-foreground/60">
                Graph
              </div>
              <h2 className="font-sans text-2xl font-semibold text-foreground md:text-3xl">Performance Graph</h2>
              {agentKey === "random" ? (
                <RandomAgentLineChart />
              ) : agentKey === "pid" ? (
                <PIDAgentLineChart />
              ) : (
                <>
                  <p className="text-sm leading-relaxed text-foreground/70">
                    Drop in a time-series chart, reward curve, or any other metric that captures how this agent behaves
                    across episodes.
                  </p>
                  <div className="rounded-2xl border border-dashed border-foreground/15 bg-background/40 p-6 text-sm text-foreground/60">
                    Graph placeholder — add your plotting component here.
                  </div>
                </>
              )}
            </div>
          </div>

          <div className="rounded-3xl border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-xl md:p-8">
            <div className="space-y-4">
              <div className="inline-flex items-center gap-2 rounded-full border border-foreground/15 bg-foreground/10 px-3 py-1 text-xs uppercase tracking-widest text-foreground/60">
                Visualization
              </div>
              <h2 className="font-sans text-2xl font-semibold text-foreground md:text-3xl">Interactive Visualization</h2>
              <p className="text-sm leading-relaxed text-foreground/70">
                Reserve this pane for WebGL canvases, 3D chambers, or any interactive view aligned with the agent’s
                control story.
              </p>
              <div className="rounded-2xl border border-dashed border-foreground/15 bg-background/40 p-6 text-sm text-foreground/60">
                Visualization placeholder — embed simulation or real-time view here.
              </div>
            </div>
          </div>
        </section>

      </div>
    </main>
  )
}