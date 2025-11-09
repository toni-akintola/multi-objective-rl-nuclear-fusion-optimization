"use client"

import { useReveal } from "@/hooks/use-reveal"
import { useState } from "react"

export function PlasmaSection() {
  const { ref: revealRef, isVisible } = useReveal(0.2)
  
  // Calculator state
  const [calcValues, setCalcValues] = useState({
    // Real tokamak parameters
    disruptionRateBaseline: 0.08,
    disruptionRateWithGuard: 0.02,
    pulsesPerYear: 1000,
    disruptionCost: 1000000,
    // Training parameters
    avgEpisodeLengthWithout: 50,
    avgEpisodeLengthWith: 150,
    numEpisodes: 10000,
    violationRateWithout: 0.15,
    violationRateWith: 0.05,
  })
  
  const [results, setResults] = useState<any>(null)
  
  const calculateCostImportance = () => {
    // Real tokamak savings
    const disruptionsWithout = calcValues.pulsesPerYear * calcValues.disruptionRateBaseline
    const disruptionsWith = calcValues.pulsesPerYear * calcValues.disruptionRateWithGuard
    const disruptionsPrevented = disruptionsWithout - disruptionsWith
    const annualCostWithout = disruptionsWithout * calcValues.disruptionCost
    const annualCostWith = disruptionsWith * calcValues.disruptionCost
    const annualSavings = annualCostWithout - annualCostWith
    const preventionRate = (disruptionsPrevented / disruptionsWithout) * 100
    const roi = (annualSavings / annualCostWith) * 100
    
    // Training cost savings
    const totalStepsWithout = calcValues.numEpisodes * calcValues.avgEpisodeLengthWithout
    const totalStepsWith = calcValues.numEpisodes * calcValues.avgEpisodeLengthWith
    const stepsPerSecond = 10
    const gpuHourCost = 2.0
    const hoursWithout = totalStepsWithout / (stepsPerSecond * 3600)
    const hoursWith = totalStepsWith / (stepsPerSecond * 3600)
    const convergenceSpeedup = 2.5
    const effectiveHoursWith = hoursWith / convergenceSpeedup
    const timeSavings = hoursWithout - effectiveHoursWith
    const trainingCostSavings = timeSavings * gpuHourCost
    const episodeLengthImprovement = ((calcValues.avgEpisodeLengthWith - calcValues.avgEpisodeLengthWithout) / calcValues.avgEpisodeLengthWithout) * 100
    
    // Violation reduction
    const violationReduction = ((calcValues.violationRateWithout - calcValues.violationRateWith) / calcValues.violationRateWithout) * 100
    
    // Importance scores
    const maxAnnualSavings = 50_000_000
    const tokamakImportance = Math.min((annualSavings / maxAnnualSavings) * 100, 100)
    const maxTrainingSavings = 100_000
    const trainingImportance = Math.min((trainingCostSavings / maxTrainingSavings) * 100, 100)
    const safetyImportance = violationReduction
    const overallImportance = tokamakImportance * 0.6 + trainingImportance * 0.2 + safetyImportance * 0.2
    
    setResults({
      tokamak: {
        disruptionsWithout,
        disruptionsWith,
        disruptionsPrevented,
        preventionRate,
        annualCostWithout,
        annualCostWith,
        annualSavings,
        roi,
      },
      training: {
        totalStepsWithout,
        totalStepsWith,
        episodeLengthImprovement,
        hoursWithout,
        hoursWith,
        convergenceSpeedup,
        timeSavings,
        trainingCostSavings,
      },
      importance: {
        overallImportance,
        tokamakImportance,
        trainingImportance,
        safetyImportance,
        costBenefitRatio: annualSavings / (hoursWith * gpuHourCost),
      },
      totalAnnualValue: annualSavings + trainingCostSavings,
    })
  }

  return (
    <section className="flex min-h-screen w-screen shrink-0 snap-start flex-col justify-center px-6 py-16 md:px-12 lg:px-16">
      <div className="mx-auto w-full max-w-4xl">
        {/* Header */}
        <div
          ref={revealRef as React.RefObject<HTMLDivElement>}
          className={`mb-12 text-center transition-all duration-700 ${
            isVisible ? "translate-y-0 opacity-100" : "translate-y-12 opacity-0"
          }`}
        >
          <div className="mb-4 inline-block rounded-full border border-foreground/20 bg-foreground/15 px-4 py-1.5 backdrop-blur-md">
            <p className="font-mono text-xs text-foreground/90">Safety Monitoring Systems</p>
          </div>
          <h1 className="mb-4 font-sans text-5xl font-light leading-tight text-foreground md:text-6xl">
            Plasma Shape Monitoring
          </h1>
          <p className="max-w-2xl mx-auto text-lg leading-relaxed text-foreground/70 md:text-xl">
            Continuous monitoring of critical plasma parameters to prevent disruptions and ensure safe operation
          </p>
        </div>

        {/* 2 Column Layout */}
        <div className="grid grid-cols-1 gap-8 md:grid-cols-2 md:gap-12">
          {/* Left Column */}
          <div className="space-y-4">
            {/* What is Shape Section */}
            <div className="rounded-xl border border-foreground/10 bg-foreground/5 p-4 backdrop-blur-md">
              <h2 className="mb-2 font-sans text-lg font-semibold text-foreground">What is Plasma Shape?</h2>
              <p className="mb-3 text-xs text-foreground/80 leading-relaxed">
                Plasma shape refers to the three-dimensional geometry and stability parameters of the confined plasma. 
                These parameters determine whether the plasma remains stable or becomes unstable, leading to disruptions.
              </p>
              
              <div className="grid gap-4">
                <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-4">
                  <div className="mb-1 font-mono text-xs font-semibold text-foreground/90">β_N</div>
                  <div className="mb-1 font-sans text-base font-medium text-foreground">Normalized Beta</div>
                  <p className="text-xs text-foreground/70">
                    Plasma pressure vs magnetic field strength. Too high → disruptions. Safe range: 0.5 - 3.0
                  </p>
                </div>
                
                <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-4">
                  <div className="mb-1 font-mono text-xs font-semibold text-foreground/90">q_min</div>
                  <div className="mb-1 font-sans text-base font-medium text-foreground">Minimum Safety Factor</div>
                  <p className="text-xs text-foreground/70">
                    Prevents internal instabilities. Too low → internal disruptions. Must be ≥ 1.0
                  </p>
                </div>
                
                <div className="rounded-lg border border-foreground/10 bg-foreground/5 p-4">
                  <div className="mb-1 font-mono text-xs font-semibold text-foreground/90">q95</div>
                  <div className="mb-1 font-sans text-base font-medium text-foreground">Edge Safety Factor</div>
                  <p className="text-xs text-foreground/70">
                    Prevents edge disruptions and ELMs. Safe range: 3.0 - 5.0
                  </p>
                </div>
              </div>
            </div>

            {/* Why Shape Matters Section */}
            <div className="rounded-xl border border-foreground/10 bg-foreground/5 p-4 backdrop-blur-md">
              <h2 className="mb-3 font-sans text-lg font-semibold text-foreground">Why Shape Monitoring is Critical</h2>
              
              <div className="space-y-2">
                <div className="flex gap-3">
                  <div className="flex h-2 w-2 shrink-0 items-center justify-center rounded-full bg-foreground/60 mt-1.5"></div>
                  <div>
                    <div className="mb-0.5 font-sans text-sm font-semibold text-foreground">Prevents Disruptions</div>
                    <p className="text-xs text-foreground/70">
                      Shape violations can lead to sudden loss of plasma confinement, causing millions in damage and weeks of downtime
                    </p>
                  </div>
                </div>
                
                <div className="flex gap-3">
                  <div className="flex h-2 w-2 shrink-0 items-center justify-center rounded-full bg-foreground/60 mt-1.5"></div>
                  <div>
                    <div className="mb-0.5 font-sans text-sm font-semibold text-foreground">Enables Self-Correction</div>
                    <p className="text-xs text-foreground/70">
                      The RL agent learns to automatically adjust coil currents when parameters drift, bringing plasma back to safe states
                    </p>
                  </div>
                </div>
                
                <div className="flex gap-3">
                  <div className="flex h-2 w-2 shrink-0 items-center justify-center rounded-full bg-foreground/60 mt-1.5"></div>
                  <div>
                    <div className="mb-0.5 font-sans text-sm font-semibold text-foreground">Massive Cost Savings</div>
                    <p className="text-xs text-foreground/70">
                      Prevents 60-80 disruptions per year, saving $60M+ annually in real tokamak operations
                    </p>
                  </div>
                </div>
                
                <div className="flex gap-3">
                  <div className="flex h-2 w-2 shrink-0 items-center justify-center rounded-full bg-foreground/60 mt-1.5"></div>
                  <div>
                    <div className="mb-0.5 font-sans text-sm font-semibold text-foreground">Training Efficiency</div>
                    <p className="text-xs text-foreground/70">
                      Episodes run 2-3x longer with shape guard, providing 3x more learning data per episode
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="space-y-4">
            {/* Cost Importance Calculator */}
            <div className="rounded-xl border border-foreground/10 bg-foreground/5 p-4 backdrop-blur-md">
              <h2 className="mb-3 font-sans text-lg font-semibold text-foreground">Cost Importance Calculator</h2>
              
              <div className="mb-4 grid gap-3 md:grid-cols-2">
                {/* Real Tokamak Parameters */}
                <div className="space-y-3 rounded-lg border border-foreground/10 bg-foreground/5 p-4">
                  <h3 className="font-sans text-sm font-semibold text-foreground">Real Tokamak Parameters</h3>
                  
                  <div>
                    <label className="mb-1 block text-xs font-medium text-foreground/80">
                      Disruption Rate (without guard) %
                    </label>
                    <input
                      type="number"
                      step="0.01"
                      value={calcValues.disruptionRateBaseline * 100}
                      onChange={(e) => setCalcValues({ ...calcValues, disruptionRateBaseline: parseFloat(e.target.value) / 100 })}
                      className="w-full rounded-lg border border-foreground/20 bg-background px-3 py-1.5 text-sm text-foreground"
                    />
                  </div>
                  
                  <div>
                    <label className="mb-1 block text-xs font-medium text-foreground/80">
                      Disruption Rate (with guard) %
                    </label>
                    <input
                      type="number"
                      step="0.01"
                      value={calcValues.disruptionRateWithGuard * 100}
                      onChange={(e) => setCalcValues({ ...calcValues, disruptionRateWithGuard: parseFloat(e.target.value) / 100 })}
                      className="w-full rounded-lg border border-foreground/20 bg-background px-3 py-1.5 text-sm text-foreground"
                    />
                  </div>
                  
                  <div>
                    <label className="mb-1 block text-xs font-medium text-foreground/80">
                      Pulses per Year
                    </label>
                    <input
                      type="number"
                      value={calcValues.pulsesPerYear}
                      onChange={(e) => setCalcValues({ ...calcValues, pulsesPerYear: parseInt(e.target.value) })}
                      className="w-full rounded-lg border border-foreground/20 bg-background px-3 py-1.5 text-sm text-foreground"
                    />
                  </div>
                  
                  <div>
                    <label className="mb-1 block text-xs font-medium text-foreground/80">
                      Cost per Disruption ($)
                    </label>
                    <input
                      type="number"
                      value={calcValues.disruptionCost}
                      onChange={(e) => setCalcValues({ ...calcValues, disruptionCost: parseFloat(e.target.value) })}
                      className="w-full rounded-lg border border-foreground/20 bg-background px-3 py-1.5 text-sm text-foreground"
                    />
                  </div>
                </div>
                
                {/* Training Parameters */}
                <div className="space-y-3 rounded-lg border border-foreground/10 bg-foreground/5 p-4">
                  <h3 className="font-sans text-sm font-semibold text-foreground">Training Parameters</h3>
                  
                  <div>
                    <label className="mb-1 block text-xs font-medium text-foreground/80">
                      Avg Episode Length (without guard)
                    </label>
                    <input
                      type="number"
                      value={calcValues.avgEpisodeLengthWithout}
                      onChange={(e) => setCalcValues({ ...calcValues, avgEpisodeLengthWithout: parseFloat(e.target.value) })}
                      className="w-full rounded-lg border border-foreground/20 bg-background px-3 py-1.5 text-sm text-foreground"
                    />
                  </div>
                  
                  <div>
                    <label className="mb-1 block text-xs font-medium text-foreground/80">
                      Avg Episode Length (with guard)
                    </label>
                    <input
                      type="number"
                      value={calcValues.avgEpisodeLengthWith}
                      onChange={(e) => setCalcValues({ ...calcValues, avgEpisodeLengthWith: parseFloat(e.target.value) })}
                      className="w-full rounded-lg border border-foreground/20 bg-background px-3 py-1.5 text-sm text-foreground"
                    />
                  </div>
                  
                  <div>
                    <label className="mb-1 block text-xs font-medium text-foreground/80">
                      Number of Episodes
                    </label>
                    <input
                      type="number"
                      value={calcValues.numEpisodes}
                      onChange={(e) => setCalcValues({ ...calcValues, numEpisodes: parseInt(e.target.value) })}
                      className="w-full rounded-lg border border-foreground/20 bg-background px-3 py-1.5 text-sm text-foreground"
                    />
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="mb-1 block text-xs font-medium text-foreground/80">
                        Violation Rate (without) %
                      </label>
                      <input
                        type="number"
                        step="0.01"
                        value={calcValues.violationRateWithout * 100}
                        onChange={(e) => setCalcValues({ ...calcValues, violationRateWithout: parseFloat(e.target.value) / 100 })}
                        className="w-full rounded-lg border border-foreground/20 bg-background px-3 py-1.5 text-sm text-foreground"
                      />
                    </div>
                    
                    <div>
                      <label className="mb-1 block text-xs font-medium text-foreground/80">
                        Violation Rate (with) %
                      </label>
                      <input
                        type="number"
                        step="0.01"
                        value={calcValues.violationRateWith * 100}
                        onChange={(e) => setCalcValues({ ...calcValues, violationRateWith: parseFloat(e.target.value) / 100 })}
                        className="w-full rounded-lg border border-foreground/20 bg-background px-3 py-1.5 text-sm text-foreground"
                      />
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="mb-4 flex justify-center">
                <button
                  onClick={calculateCostImportance}
                  className="rounded-lg bg-blue-600 px-6 py-2 text-sm font-mono text-white transition-all hover:bg-blue-700 hover:scale-105"
                >
                  Calculate Cost Importance
                </button>
              </div>
              
              {/* Results */}
              {results && (
                <div className="space-y-4 rounded-lg border border-foreground/10 bg-gradient-to-br from-green-500/10 to-blue-500/10 p-4">
                  <h3 className="font-sans text-base font-semibold text-foreground">Results</h3>
                  
                  {/* Real Tokamak Savings */}
                  <div className="rounded-lg bg-foreground/5 p-3">
                    <h4 className="mb-2 font-sans text-sm font-semibold text-foreground">Real Tokamak Savings</h4>
                    <div className="grid gap-2 text-xs">
                      <div>
                        <span className="text-foreground/60">Disruptions prevented:</span>
                        <span className="ml-2 font-mono font-semibold text-foreground">{results.tokamak.disruptionsPrevented.toFixed(1)}</span>
                      </div>
                      <div>
                        <span className="text-foreground/60">Prevention rate:</span>
                        <span className="ml-2 font-mono font-semibold text-foreground">{results.tokamak.preventionRate.toFixed(1)}%</span>
                      </div>
                      <div>
                        <span className="text-foreground/60">Annual savings:</span>
                        <span className="ml-2 font-mono font-semibold text-foreground">${(results.tokamak.annualSavings / 1_000_000).toFixed(1)}M</span>
                      </div>
                      <div>
                        <span className="text-foreground/60">ROI:</span>
                        <span className="ml-2 font-mono font-semibold text-foreground">{results.tokamak.roi.toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                  
                  {/* Training Savings */}
                  <div className="rounded-lg bg-foreground/5 p-3">
                    <h4 className="mb-2 font-sans text-sm font-semibold text-foreground">Training Cost Savings</h4>
                    <div className="grid gap-2 text-xs">
                      <div>
                        <span className="text-foreground/60">Episode length improvement:</span>
                        <span className="ml-2 font-mono font-semibold text-foreground">+{results.training.episodeLengthImprovement.toFixed(1)}%</span>
                      </div>
                      <div>
                        <span className="text-foreground/60">Convergence speedup:</span>
                        <span className="ml-2 font-mono font-semibold text-foreground">{results.training.convergenceSpeedup.toFixed(1)}x</span>
                      </div>
                      <div>
                        <span className="text-foreground/60">Time savings:</span>
                        <span className="ml-2 font-mono font-semibold text-foreground">{results.training.timeSavings.toFixed(1)} hours</span>
                      </div>
                      <div>
                        <span className="text-foreground/60">Training cost savings:</span>
                        <span className="ml-2 font-mono font-semibold text-foreground">${results.training.trainingCostSavings.toFixed(2)}</span>
                      </div>
                    </div>
                  </div>
                  
                  {/* Importance Scores */}
                  <div className="rounded-lg bg-foreground/5 p-3">
                    <h4 className="mb-2 font-sans text-sm font-semibold text-foreground">Importance Scores</h4>
                    <div className="grid gap-2 text-xs">
                      <div>
                        <span className="text-foreground/60">Overall importance:</span>
                        <span className="ml-2 font-mono text-base font-bold text-foreground">{results.importance.overallImportance.toFixed(1)}/100</span>
                      </div>
                      <div>
                        <span className="text-foreground/60">Tokamak importance:</span>
                        <span className="ml-2 font-mono font-semibold text-foreground">{results.importance.tokamakImportance.toFixed(1)}/100</span>
                      </div>
                      <div>
                        <span className="text-foreground/60">Training importance:</span>
                        <span className="ml-2 font-mono font-semibold text-foreground">{results.importance.trainingImportance.toFixed(1)}/100</span>
                      </div>
                      <div>
                        <span className="text-foreground/60">Safety importance:</span>
                        <span className="ml-2 font-mono font-semibold text-foreground">{results.importance.safetyImportance.toFixed(1)}/100</span>
                      </div>
                    </div>
                  </div>
                  
                  {/* Total Value */}
                  <div className="rounded-lg border-2 border-green-500/30 bg-green-500/10 p-3">
                    <div className="text-center">
                      <div className="mb-1 text-xs text-foreground/60">Total Annual Value</div>
                      <div className="font-mono text-2xl font-bold text-foreground">
                        ${(results.totalAnnualValue / 1_000_000).toFixed(2)}M
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Live Visualization */}
            <div className="flex flex-col items-center gap-4 rounded-xl border border-foreground/10 bg-foreground/5 p-4 backdrop-blur-md">
              <h2 className="font-sans text-lg font-semibold text-foreground">Live Visualization</h2>
              <p className="text-center text-xs text-foreground/70">
                Watch the shape guard system in action. See how plasma parameters are monitored in real-time 
                and how the system self-corrects when violations occur.
              </p>
              <button
                onClick={async () => {
                  try {
                    const response = await fetch("/api/launch-python")
                    const data = await response.json()
                    if (data.success) {
                      alert("Visualization launched! A matplotlib window should open.")
                    } else {
                      alert(`Error: ${data.error}`)
                    }
                  } catch (error) {
                    alert("Failed to launch visualization. Please ensure dependencies are installed.")
                  }
                }}
                className="rounded-lg bg-blue-600 px-6 py-2 text-sm font-mono text-white transition-all hover:bg-blue-700 hover:scale-105"
              >
                Launch Visualization
              </button>
              <p className="text-center font-mono text-xs text-foreground/50">
                Opens a real-time window showing plasma shape monitoring
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

