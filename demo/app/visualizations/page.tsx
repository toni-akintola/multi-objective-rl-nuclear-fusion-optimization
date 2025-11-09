"use client"

import { useMemo, useRef, useEffect, useState } from "react"
import Link from "next/link"
import { Shader, Swirl, ChromaFlow } from "shaders/react"
import { CustomCursor } from "@/components/custom-cursor"
import { GrainOverlay } from "@/components/grain-overlay"
import { MagneticButton } from "@/components/magnetic-button"

const SECTION_CONFIG = [
  { slug: "sac", title: "SAC Agent", description: "High-level snapshots of plasma control experiments." },
  { slug: "pid", title: "PID Agent", description: "Placeholder for deterministic baselines and controller diagnostics." },
  { slug: "random", title: "Random Agent", description: "Space reserved for policy rollouts and reward traces." },
]

export default function VisualizationsPage() {
  const sections = useMemo(() => SECTION_CONFIG, [])
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
        <div className="hidden items-center gap-8 md:flex">
          {sections.map((section) => (
            <Link key={section.slug} href={`/visualizations/${section.slug}`} className="font-sans text-sm font-medium text-foreground/80 transition-colors hover:text-foreground">
              {section.title}
            </Link>
          ))}
        </div>
        <MagneticButton size="default" variant="secondary" onClick={() => window.history.back()}>
          Back
        </MagneticButton>
      </nav>

      <div
        className={`relative z-10 mt-32 flex flex-col gap-24 px-6 pb-24 transition-opacity duration-700 md:px-12 lg:px-24 ${
          isLoaded ? "opacity-100" : "opacity-0"
        }`}
      >
        <section className="flex min-h-[60vh] flex-col justify-end gap-8 pt-12">
          <div className="max-w-3xl space-y-6">
            <div className="inline-flex items-center gap-2 rounded-full border border-foreground/15 bg-foreground/10 px-4 py-1.5 backdrop-blur-md">
              <span className="font-mono text-xs uppercase tracking-widest text-foreground/70">Visualizations Lab</span>
            </div>
            <h1 className="font-sans text-6xl font-light leading-[1.05] tracking-tight md:text-7xl lg:text-8xl">
              A dedicated space for plasma control storytelling.
            </h1>
            <p className="max-w-2xl text-lg leading-relaxed text-foreground/80 md:text-xl">
              This page mirrors the immersive feel of our homepage while leaving room to stage interactive plots,
              diagnostics, and live experiment dashboards. Sections below are ready to host upcoming content.
            </p>
          </div>
        </section>

        <div className="grid gap-12 md:grid-cols-2">
          {sections.map((section) => (
            <Link
              key={section.slug}
              href={`/visualizations/${section.slug}`}
              className="group relative block min-h-[32vh] rounded-3xl border border-foreground/10 bg-foreground/5 p-8 transition-transform duration-300 backdrop-blur-xl hover:-translate-y-2 hover:border-foreground/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-foreground/40 md:p-12"
            >
              <div className="flex h-full flex-col justify-between gap-6">
                <div className="space-y-4">
                  <div className="inline-flex items-center gap-2 rounded-full border border-foreground/15 bg-foreground/10 px-3 py-1 text-xs uppercase tracking-widest text-foreground/60">
                    {section.slug.toUpperCase()}
                  </div>
                  <h2 className="font-sans text-3xl font-semibold text-foreground transition-colors duration-200 group-hover:text-foreground md:text-4xl">
                    {section.title}
                  </h2>
                  <p className="max-w-xl text-base leading-relaxed text-foreground/75">{section.description}</p>
                </div>
                <div className="flex items-center justify-between text-sm font-medium text-foreground/70">
                  <span>Open visualization workspace</span>
                  <span className="transition-transform duration-300 group-hover:translate-x-2">→</span>
                </div>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </main>
  )
}

