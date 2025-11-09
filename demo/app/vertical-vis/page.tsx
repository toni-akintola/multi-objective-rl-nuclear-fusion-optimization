"use client"

import dynamic from "next/dynamic"
import { CustomCursor } from "@/components/custom-cursor"
import { GrainOverlay } from "@/components/grain-overlay"
import { useRef, useEffect, useState } from "react"
import Link from "next/link"
import { VerticalPython } from "@/components/vertical-python"

// Dynamically import shader components to avoid SSR issues
const ShaderBackground = dynamic(
  () => import("shaders/react").then((mod) => {
    const { Shader, ChromaFlow, Swirl } = mod
    return function ShaderBackgroundComponent() {
      return (
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
      )
    }
  }),
  { ssr: false }
)

export default function VerticalVisPage() {
  const [isLoaded, setIsLoaded] = useState(false)
  const shaderContainerRef = useRef<HTMLDivElement>(null)

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
    <main className="relative min-h-screen w-full overflow-x-hidden bg-background">
      <CustomCursor />
      <GrainOverlay />

      <div
        ref={shaderContainerRef}
        className={`fixed inset-0 z-0 transition-opacity duration-700 ${isLoaded ? "opacity-100" : "opacity-0"}`}
        style={{ contain: "strict" }}
      >
        <ShaderBackground />
      </div>

      <nav
        className={`fixed left-0 right-0 top-0 z-50 flex items-center justify-between px-6 py-6 transition-opacity duration-700 md:px-12 ${
          isLoaded ? "opacity-100" : "opacity-0"
        }`}
      >
        <Link
          href="/fusion"
          className="flex items-center gap-2 transition-transform hover:scale-105"
        >
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/15 backdrop-blur-md transition-all duration-300 hover:scale-110 hover:bg-foreground/25">
            <span className="font-sans text-xl font-bold text-foreground">ϕ</span>
          </div>
          <span className="font-sans text-xl font-semibold tracking-tight text-foreground">Fusion Lab</span>
        </Link>

        <div className="hidden items-center gap-8 md:flex">
          {["Problem", "Solution", "Approach", "Plasma", "Vertical", "Insights"].map((item) => {
            // Vertical is active on this page
            if (item === "Vertical") {
              return (
                <button
                  key={item}
                  className="group relative font-sans text-sm font-medium transition-colors text-foreground"
                >
                  {item}
                  <span className="absolute -bottom-1 left-0 h-px bg-foreground transition-all duration-300 w-full" />
                </button>
              )
            }
            // Plasma links to chamber page
            if (item === "Plasma") {
              return (
                <Link
                  key={item}
                  href="/chamber"
                  className="group relative font-sans text-sm font-medium transition-colors text-foreground/80 hover:text-foreground"
                >
                  {item}
                  <span className="absolute -bottom-1 left-0 h-px bg-foreground transition-all duration-300 w-0 group-hover:w-full" />
                </Link>
              )
            }
            // Map to fusion page sections
            const sectionMap: Record<string, number> = {
              "Problem": 1,
              "Solution": 2,
              "Approach": 3,
              "Insights": 3
            }
            const sectionIndex = sectionMap[item] ?? 0
            return (
              <Link
                key={item}
                href={`/fusion?section=${sectionIndex}`}
                className="group relative font-sans text-sm font-medium transition-colors text-foreground/80 hover:text-foreground"
              >
                {item}
                <span className="absolute -bottom-1 left-0 h-px bg-foreground transition-all duration-300 w-0 group-hover:w-full" />
              </Link>
            )
          })}
        </div>
      </nav>

      <div
        className={`relative z-10 flex min-h-screen flex-col items-center px-6 pt-32 pb-16 transition-opacity duration-700 md:px-12 ${
          isLoaded ? "opacity-100" : "opacity-0"
        }`}
      >
        <div className="w-full max-w-4xl">
          {/* Hero Section */}
          <div className="mb-12 text-center">
            <div className="mb-4 inline-block rounded-full border border-foreground/20 bg-foreground/15 px-4 py-1.5 backdrop-blur-md">
              <p className="font-mono text-xs text-foreground/90">Safety Monitoring Systems</p>
            </div>
            <h1 className="mb-4 font-sans text-5xl font-light leading-tight text-foreground md:text-6xl">
              Vertical Position Monitoring
            </h1>
            <p className="max-w-2xl mx-auto text-lg leading-relaxed text-foreground/70 md:text-xl">
              Continuous monitoring of plasma vertical position to prevent Vertical Displacement Events and ensure safe operation
            </p>
          </div>

          {/* What is Vertical Position? */}
          <div className="mb-12 rounded-2xl border border-foreground/10 bg-foreground/5 p-8 backdrop-blur-md">
            <h2 className="mb-4 font-sans text-2xl font-semibold text-foreground">What is Vertical Position?</h2>
            <p className="mb-6 text-foreground/80 leading-relaxed">
              Vertical position (Z) refers to the height of the plasma center within the tokamak chamber. 
              When the plasma drifts too far up or down, it can hit the chamber walls in milliseconds, causing 
              a Vertical Displacement Event (VDE) - one of the most dangerous and costly disruptions in fusion.
            </p>
          </div>

          {/* Parameters Section */}
          <div className="mb-12">
            <div className="grid gap-6 md:grid-cols-2">
              <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-md">
                <div className="mb-2 font-mono text-sm font-semibold text-foreground/90">Z Position</div>
                <div className="mb-2 font-sans text-lg font-medium text-foreground">Vertical Location</div>
                <p className="text-sm text-foreground/70">
                  Height of plasma center in the tokamak. Safe range: -5cm to +5cm from magnetic axis
                </p>
              </div>
              <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-md">
                <div className="mb-2 font-mono text-sm font-semibold text-foreground/90">dZ/dt</div>
                <div className="mb-2 font-sans text-lg font-medium text-foreground">Vertical Velocity</div>
                <p className="text-sm text-foreground/70">
                  How fast the plasma is moving up or down. Must be kept below 0.5 cm/step
                </p>
              </div>
            </div>
          </div>

          {/* Why Monitoring is Critical */}
          <div className="mb-12">
            <h2 className="mb-6 text-center font-sans text-2xl font-semibold text-foreground">
              Why Vertical Monitoring is Critical
            </h2>
            <div className="grid gap-6 md:grid-cols-2">
              <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-md">
                <h3 className="mb-2 font-sans text-lg font-semibold text-foreground">
                  Prevents VDEs (Vertical Displacement Events)
                </h3>
                <p className="text-sm text-foreground/70">
                  When plasma moves too far up or down, it can hit the chamber walls in milliseconds. This causes instant plasma loss and severe damage costing $5M+ and 3-6 months of downtime.
                </p>
              </div>
              <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-md">
                <h3 className="mb-2 font-sans text-lg font-semibold text-foreground">
                  Enables Real-Time Correction
                </h3>
                <p className="text-sm text-foreground/70">
                  The RL agent learns to detect early signs of vertical instability and automatically adjusts the vertical field coils to bring the plasma back to center before a VDE occurs.
                </p>
              </div>
              <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-md">
                <h3 className="mb-2 font-sans text-lg font-semibold text-foreground">
                  Massive Cost Savings
                </h3>
                <p className="text-sm text-foreground/70">
                  With vertical guard enabled, VDE rates drop by 95%, episodes run 3x longer, and tokamaks save $50M+ annually in avoided repairs and downtime.
                </p>
              </div>
              <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-md">
                <h3 className="mb-2 font-sans text-lg font-semibold text-foreground">
                  Training Efficiency
                </h3>
                <p className="text-sm text-foreground/70">
                  Episodes run 3-4x longer with vertical guard, providing significantly more learning data per episode and faster convergence to safe policies.
                </p>
              </div>
            </div>
          </div>

          {/* Visualization Section */}
          <div className="mb-12">
            <h2 className="mb-6 text-center font-sans text-2xl font-semibold text-foreground">
              Live Visualization
            </h2>
            <div className="text-center">
              <p className="mb-8 text-lg text-foreground/70">
                Watch the vertical guard system in action. See how plasma vertical position is monitored in real-time and how the system self-corrects when drift occurs.
              </p>
              <VerticalPython />
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}

