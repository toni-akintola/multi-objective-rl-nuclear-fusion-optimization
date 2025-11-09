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
  
  // Calculator state
  const [calcValues, setCalcValues] = useState({
    // Real tokamak parameters
    vdeRateBaseline: 0.10,
    vdeRateWithGuard: 0.01,
    pulsesPerYear: 1000,
    vdeCost: 5000000,
    // Training parameters
    avgEpisodeLengthWithout: 50,
    avgEpisodeLengthWith: 180,
    numEpisodes: 10000,
    vdeRateTrainingWithout: 0.20,
    vdeRateTrainingWith: 0.03,
  })
  
  const [results, setResults] = useState<any>(null)
  
  const calculateVDEImportance = () => {
    // Real tokamak savings
    const vdesWithout = calcValues.pulsesPerYear * calcValues.vdeRateBaseline
    const vdesWith = calcValues.pulsesPerYear * calcValues.vdeRateWithGuard
    const vdesPrevented = vdesWithout - vdesWith
    const annualCostWithout = vdesWithout * calcValues.vdeCost
    const annualCostWith = vdesWith * calcValues.vdeCost
    const annualSavings = annualCostWithout - annualCostWith
    const preventionRate = (vdesPrevented / vdesWithout) * 100
    const roi = (annualSavings / annualCostWith) * 100
    
    // Training cost savings
    const totalStepsWithout = calcValues.numEpisodes * calcValues.avgEpisodeLengthWithout
    const totalStepsWith = calcValues.numEpisodes * calcValues.avgEpisodeLengthWith
    const stepsPerSecond = 10
    const gpuHourCost = 2.0
    const hoursWithout = totalStepsWithout / (stepsPerSecond * 3600)
    const hoursWith = totalStepsWith / (stepsPerSecond * 3600)
    const convergenceSpeedup = 2.0
    const effectiveHoursWith = hoursWith / convergenceSpeedup
    const timeSavings = hoursWithout - effectiveHoursWith
    const trainingCostSavings = timeSavings * gpuHourCost
    const episodeLengthImprovement = ((calcValues.avgEpisodeLengthWith - calcValues.avgEpisodeLengthWithout) / calcValues.avgEpisodeLengthWithout) * 100
    
    // VDE reduction
    const vdeReduction = ((calcValues.vdeRateTrainingWithout - calcValues.vdeRateTrainingWith) / calcValues.vdeRateTrainingWithout) * 100
    
    // Importance scores
    const maxAnnualSavings = 50_000_000
    const tokamakImportance = Math.min((annualSavings / maxAnnualSavings) * 100, 100)
    const maxTrainingSavings = 100_000
    const trainingImportance = Math.min((trainingCostSavings / maxTrainingSavings) * 100, 100)
    const overallImportance = (tokamakImportance + trainingImportance) / 2
    
    setResults({
      // Real tokamak
      vdesWithout,
      vdesWith,
      vdesPrevented,
      annualCostWithout,
      annualCostWith,
      annualSavings,
      preventionRate,
      roi,
      // Training
      totalStepsWithout,
      totalStepsWith,
      hoursWithout,
      effectiveHoursWith,
      timeSavings,
      trainingCostSavings,
      episodeLengthImprovement,
      vdeReduction,
      // Importance
      tokamakImportance,
      trainingImportance,
      overallImportance,
    })
  }

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
        <div className="absolute inset-0 bg-black/40" />
      </div>

      <nav
        className={`fixed left-0 right-0 top-0 z-50 flex items-center justify-between border-b border-foreground/10 bg-background/80 px-6 py-4 backdrop-blur-md transition-opacity duration-700 md:px-12 ${
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

      <div className={`relative z-10 px-6 pb-24 pt-32 transition-opacity duration-700 md:px-12 ${isLoaded ? "opacity-100" : "opacity-0"}`}>
        <div className="mx-auto max-w-7xl">
          {/* Hero Section */}
          <div className="mb-16 text-center">
            <div className="mb-4 inline-block rounded-full border border-foreground/20 bg-foreground/10 px-4 py-1.5 backdrop-blur-md">
              <p className="font-mono text-xs text-foreground/90">Safety Monitoring Systems</p>
            </div>
            <h1 className="mb-6 font-sans text-5xl font-light leading-tight tracking-tight text-foreground md:text-6xl lg:text-7xl">
              Vertical Position Monitoring
            </h1>
            <p className="mx-auto mb-8 max-w-2xl text-lg leading-relaxed text-foreground/80 md:text-xl">
              Continuous monitoring of plasma vertical position to prevent Vertical Displacement Events and ensure safe operation
            </p>
          </div>

          {/* What is Vertical Position? */}
          <div className="mb-16">
            <h2 className="mb-8 text-center font-sans text-3xl font-light text-foreground md:text-4xl">
              What is Vertical Position?
            </h2>
            <div className="mx-auto max-w-4xl rounded-2xl border border-foreground/10 bg-foreground/5 p-8 backdrop-blur-md">
              <p className="mb-6 text-lg leading-relaxed text-foreground/90">
                Vertical position (Z) refers to the height of the plasma center within the tokamak chamber. 
                When the plasma drifts too far up or down, it can hit the chamber walls in milliseconds, causing 
                a Vertical Displacement Event (VDE) - one of the most dangerous and costly disruptions in fusion.
              </p>
            </div>
          </div>

          {/* Parameters Section */}
          <div className="mb-16">
            <div className="grid gap-6 md:grid-cols-2">
              <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-md">
                <div className="mb-2 font-mono text-sm font-semibold text-blue-400">Z Position</div>
                <div className="mb-2 font-sans text-lg font-medium text-foreground">Vertical Location</div>
                <p className="text-sm text-foreground/70">
                  Height of plasma center in the tokamak. Safe range: -5cm to +5cm from magnetic axis
                </p>
              </div>
              <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-6 backdrop-blur-md">
                <div className="mb-2 font-mono text-sm font-semibold text-green-400">dZ/dt</div>
                <div className="mb-2 font-sans text-lg font-medium text-foreground">Vertical Velocity</div>
                <p className="text-sm text-foreground/70">
                  How fast the plasma is moving up or down. Must be kept below 0.5 cm/step
                </p>
              </div>
            </div>
          </div>

          {/* Why Monitoring is Critical */}
          <div className="mb-16">
            <h2 className="mb-8 text-center font-sans text-3xl font-light text-foreground md:text-4xl">
              Why Vertical Monitoring is Critical
            </h2>
            <div className="space-y-4">
              <div className="flex gap-4">
                <div className="flex h-2 w-2 shrink-0 items-center justify-center rounded-full bg-foreground/60 mt-2"></div>
                <div>
                  <h3 className="mb-2 font-sans text-xl font-semibold text-foreground">
                    Prevents VDEs (Vertical Displacement Events)
                  </h3>
                  <p className="text-foreground/70">
                    When plasma moves too far up or down, it can hit the chamber walls in milliseconds. This causes instant plasma loss and severe damage costing $5M+ and 3-6 months of downtime.
                  </p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="flex h-2 w-2 shrink-0 items-center justify-center rounded-full bg-foreground/60 mt-2"></div>
                <div>
                  <h3 className="mb-2 font-sans text-xl font-semibold text-foreground">
                    Enables Real-Time Correction
                  </h3>
                  <p className="text-foreground/70">
                    The RL agent learns to detect early signs of vertical instability and automatically adjusts the vertical field coils to bring the plasma back to center before a VDE occurs.
                  </p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="flex h-2 w-2 shrink-0 items-center justify-center rounded-full bg-foreground/60 mt-2"></div>
                <div>
                  <h3 className="mb-2 font-sans text-xl font-semibold text-foreground">
                    Massive Cost Savings
                  </h3>
                  <p className="text-foreground/70">
                    With vertical guard enabled, VDE rates drop by 95%, episodes run 3x longer, and tokamaks save $50M+ annually in avoided repairs and downtime.
                  </p>
                </div>
              </div>
              <div className="flex gap-4">
                <div className="flex h-2 w-2 shrink-0 items-center justify-center rounded-full bg-foreground/60 mt-2"></div>
                <div>
                  <h3 className="mb-2 font-sans text-xl font-semibold text-foreground">
                    Training Efficiency
                  </h3>
                  <p className="text-foreground/70">
                    Episodes run 3-4x longer with vertical guard, providing significantly more learning data per episode and faster convergence to safe policies.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Calculator Section */}
          <div className="mb-16 rounded-2xl border border-foreground/10 bg-foreground/5 p-8 backdrop-blur-md">
            <h2 className="mb-8 text-center font-sans text-3xl font-light text-foreground md:text-4xl">
              Cost Importance Calculator
            </h2>
            
            <div className="grid gap-8 lg:grid-cols-2">
              {/* Inputs */}
              <div className="space-y-6">
                <h3 className="mb-4 font-sans text-lg font-semibold text-foreground">Real Tokamak Parameters</h3>
                
                <div>
                  <label className="mb-2 block text-sm text-foreground/70">VDE Rate (Baseline) (%)</label>
                  <input
                    type="number"
                    value={calcValues.vdeRateBaseline * 100}
                    onChange={(e) => setCalcValues({...calcValues, vdeRateBaseline: parseFloat(e.target.value) / 100})}
                    className="w-full rounded-lg border border-foreground/20 bg-foreground/10 px-4 py-2 text-foreground backdrop-blur-md"
                    step="0.1"
                  />
                </div>

                <div>
                  <label className="mb-2 block text-sm text-foreground/70">VDE Rate (With Guard) (%)</label>
                  <input
                    type="number"
                    value={calcValues.vdeRateWithGuard * 100}
                    onChange={(e) => setCalcValues({...calcValues, vdeRateWithGuard: parseFloat(e.target.value) / 100})}
                    className="w-full rounded-lg border border-foreground/20 bg-foreground/10 px-4 py-2 text-foreground backdrop-blur-md"
                    step="0.1"
                  />
                </div>

                <div>
                  <label className="mb-2 block text-sm text-foreground/70">Pulses per Year</label>
                  <input
                    type="number"
                    value={calcValues.pulsesPerYear}
                    onChange={(e) => setCalcValues({...calcValues, pulsesPerYear: parseInt(e.target.value)})}
                    className="w-full rounded-lg border border-foreground/20 bg-foreground/10 px-4 py-2 text-foreground backdrop-blur-md"
                  />
                </div>

                <div>
                  <label className="mb-2 block text-sm text-foreground/70">Cost per VDE ($)</label>
                  <input
                    type="number"
                    value={calcValues.vdeCost}
                    onChange={(e) => setCalcValues({...calcValues, vdeCost: parseInt(e.target.value)})}
                    className="w-full rounded-lg border border-foreground/20 bg-foreground/10 px-4 py-2 text-foreground backdrop-blur-md"
                    step="100000"
                  />
                </div>

                <h3 className="mb-4 mt-6 font-sans text-lg font-semibold text-foreground">Training Parameters</h3>

                <div>
                  <label className="mb-2 block text-sm text-foreground/70">Avg Episode Length (without guard)</label>
                  <input
                    type="number"
                    value={calcValues.avgEpisodeLengthWithout}
                    onChange={(e) => setCalcValues({...calcValues, avgEpisodeLengthWithout: parseInt(e.target.value)})}
                    className="w-full rounded-lg border border-foreground/20 bg-foreground/10 px-4 py-2 text-foreground backdrop-blur-md"
                  />
                </div>

                <div>
                  <label className="mb-2 block text-sm text-foreground/70">Avg Episode Length (with guard)</label>
                  <input
                    type="number"
                    value={calcValues.avgEpisodeLengthWith}
                    onChange={(e) => setCalcValues({...calcValues, avgEpisodeLengthWith: parseInt(e.target.value)})}
                    className="w-full rounded-lg border border-foreground/20 bg-foreground/10 px-4 py-2 text-foreground backdrop-blur-md"
                  />
                </div>

                <div>
                  <label className="mb-2 block text-sm text-foreground/70">Number of Episodes</label>
                  <input
                    type="number"
                    value={calcValues.numEpisodes}
                    onChange={(e) => setCalcValues({...calcValues, numEpisodes: parseInt(e.target.value)})}
                    className="w-full rounded-lg border border-foreground/20 bg-foreground/10 px-4 py-2 text-foreground backdrop-blur-md"
                  />
                </div>

                <div>
                  <label className="mb-2 block text-sm text-foreground/70">VDE Rate (training without) %</label>
                  <input
                    type="number"
                    value={calcValues.vdeRateTrainingWithout * 100}
                    onChange={(e) => setCalcValues({...calcValues, vdeRateTrainingWithout: parseFloat(e.target.value) / 100})}
                    className="w-full rounded-lg border border-foreground/20 bg-foreground/10 px-4 py-2 text-foreground backdrop-blur-md"
                    step="0.1"
                  />
                </div>

                <div>
                  <label className="mb-2 block text-sm text-foreground/70">VDE Rate (training with) %</label>
                  <input
                    type="number"
                    value={calcValues.vdeRateTrainingWith * 100}
                    onChange={(e) => setCalcValues({...calcValues, vdeRateTrainingWith: parseFloat(e.target.value) / 100})}
                    className="w-full rounded-lg border border-foreground/20 bg-foreground/10 px-4 py-2 text-foreground backdrop-blur-md"
                    step="0.1"
                  />
                </div>

                <button
                  onClick={calculateVDEImportance}
                  className="w-full rounded-lg bg-blue-600 px-6 py-3 font-semibold text-white transition-all hover:bg-blue-700 hover:scale-105"
                >
                  Calculate Cost Importance
                </button>
              </div>

              {/* Results */}
              <div className="space-y-6">
                <h3 className="mb-4 font-sans text-lg font-semibold text-foreground">Results</h3>
                
                {results ? (
                  <div className="space-y-4">
                    <div className="rounded-lg border border-green-500/20 bg-green-500/10 p-4">
                      <div className="text-sm text-foreground/70">Annual Savings</div>
                      <div className="text-2xl font-bold text-green-400">
                        ${results.annualSavings.toLocaleString()}
                      </div>
                    </div>

                    <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-4">
                      <div className="text-sm text-foreground/70">VDEs Prevented per Year</div>
                      <div className="text-xl font-semibold text-foreground">
                        {results.vdesPrevented.toFixed(0)} VDEs
                      </div>
                    </div>

                    <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-4">
                      <div className="text-sm text-foreground/70">Prevention Rate</div>
                      <div className="text-xl font-semibold text-foreground">
                        {results.preventionRate.toFixed(1)}%
                      </div>
                    </div>

                    <div className="rounded-lg border border-blue-500/20 bg-blue-500/10 p-4">
                      <div className="text-sm text-foreground/70">Overall Importance Score</div>
                      <div className="text-2xl font-bold text-blue-400">
                        {results.overallImportance.toFixed(0)}/100
                      </div>
                    </div>

                    <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-4">
                      <div className="text-sm text-foreground/70">Episode Length Improvement</div>
                      <div className="text-xl font-semibold text-foreground">
                        +{results.episodeLengthImprovement.toFixed(1)}%
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center rounded-lg border border-foreground/10 bg-foreground/5 p-12 text-foreground/50">
                    Click "Calculate Impact" to see results
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Visualization Section */}
          <div className="mb-16">
            <h2 className="mb-8 text-center font-sans text-3xl font-light text-foreground md:text-4xl">
              Live Visualization
            </h2>
            <div className="mx-auto max-w-4xl text-center">
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

