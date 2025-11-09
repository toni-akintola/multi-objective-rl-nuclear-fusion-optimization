"use client"

import { useRouter } from "next/navigation"
import { MagneticButton } from "@/components/magnetic-button"
import { Tokamak3DR3F } from "@/components/tokamak-3d-r3f"
import { useState } from "react"

export function FusionHeroSection() {
  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState<{ 
    reward: number; 
    numSteps?: number;
    rewards?: number[];
    observation: number[] | any;
    observation_raw?: any;
    action?: any;
    episode_step?: number;
  } | null>(null)
  const [currentStep, setCurrentStep] = useState(0)

  const handleRunInference = async () => {
    console.log("🔵 [FRONTEND] Run Inference button clicked")
    setIsRunning(true)
    setResult(null)
    setCurrentStep(0)
    
    const startTime = Date.now()
    const numSteps = 300
    
    try {
      // First reset the environment
      console.log("📡 [FRONTEND] Resetting environment...")
      const resetResponse = await fetch("http://localhost:8000/reset", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      })
      
      if (!resetResponse.ok) {
        throw new Error(`Reset failed: ${resetResponse.status}`)
      }
      
      const resetData = await resetResponse.json()
      console.log("✅ [FRONTEND] Environment reset")
      
      // Set initial observation
      setResult({
        reward: 0,
        numSteps: 0,
        rewards: [],
        observation: resetData.observation,
        observation_raw: resetData.observation_raw,
        action: null,
        episode_step: resetData.episode_step || 0,
      })
      setCurrentStep(0)
      
      // Run steps one by one to see updates
      let cumulativeReward = 0
      const rewards: number[] = []
      
      for (let step = 0; step < numSteps; step++) {
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), 5000) // 5 second timeout per step
        
        try {
          const response = await fetch(`http://localhost:8000/step?deterministic=true&num_steps=1`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            signal: controller.signal,
          })
          
          clearTimeout(timeoutId)
          
          if (!response.ok) {
            console.error(`❌ [FRONTEND] HTTP error! status: ${response.status}`)
            throw new Error(`HTTP error! status: ${response.status}`)
          }
          
          const data = await response.json()
          cumulativeReward += data.reward
          rewards.push(data.reward)
          
          // Update visualization with each step
          setResult({
            reward: cumulativeReward,
            numSteps: step + 1,
            rewards: [...rewards],
            observation: data.observation,
            observation_raw: data.observation_raw,
            action: data.action,
            episode_step: data.episode_step,
          })
          setCurrentStep(step + 1)
          
          // Small delay to make updates visible
          await new Promise(resolve => setTimeout(resolve, 50))
          
          // Stop if episode ended
          if (data.terminated || data.truncated) {
            console.log(`⚠️ [FRONTEND] Episode ended at step ${step + 1}`)
            break
          }
        } catch (error: any) {
          if (error.name === 'AbortError') {
            console.error(`❌ [FRONTEND] Step ${step + 1} timed out`)
            break
          }
          throw error
        }
      }
      
      const requestTime = Date.now() - startTime
      console.log(`⏱️ [FRONTEND] Completed ${currentStep} steps in ${requestTime}ms`)
      
    } catch (error: any) {
      console.error("❌ [FRONTEND] Error running inference:", error)
      if (error.name === 'AbortError') {
        alert("Request timed out. The API server may not be responding. Check if it's running on http://localhost:8000")
      } else if (error.message?.includes('Failed to fetch') || error.message?.includes('NetworkError')) {
        alert("Cannot connect to API server. Make sure it's running on http://localhost:8000")
      } else {
        alert(`Failed to run inference: ${error.message || error}`)
      }
    } finally {
      setIsRunning(false)
      console.log("🏁 [FRONTEND] Inference request finished")
    }
  }

  const router = useRouter()

  return (
    <section className="flex min-h-screen w-screen shrink-0 flex-col items-center justify-start px-6 pt-24 pb-24 md:px-12 md:pt-32 md:pb-32 lg:pt-36 lg:pb-36">
      <div className="max-w-3xl w-full text-center">
        <h1 className="mb-6 animate-in fade-in slide-in-from-bottom-8 font-sans text-6xl font-light leading-[1.1] tracking-tight text-foreground duration-1000 md:text-7xl lg:text-8xl">
          <span className="text-balance">
            Controlling
            <br />
            limitless energy
          </span>
        </h1>
        <p className="mb-8 mx-auto max-w-xl animate-in fade-in slide-in-from-bottom-4 text-lg leading-relaxed text-foreground/80 duration-1000 delay-200 md:text-xl">
          <span className="text-pretty">
            Harnessing reinforcement learning to master plasma dynamics and unlock the potential of fusion energy.
          </span>
        </p>
        <div className="flex animate-in fade-in slide-in-from-bottom-4 flex-col gap-4 duration-1000 delay-300 sm:flex-row sm:items-center sm:justify-center">
          <MagneticButton size="lg" variant="primary" onClick={() => router.push("/visualizations/sac")}>
            Visualizations
            
          </MagneticButton>
          <MagneticButton 
            size="lg" 
            variant="secondary" 
            onClick={handleRunInference}
            disabled={isRunning}
          >
            {isRunning ? "Running..." : "Run Inference"}
          </MagneticButton>
        </div>
        
        {result && (
          <div className="mt-6 mx-auto max-w-4xl space-y-4">
            {/* React Three Fiber 3D Visualization */}
            <div className="flex justify-center">
              <Tokamak3DR3F 
                observation={result.observation_raw || null}
                action={result.action}
                step={result.episode_step}
              />
            </div>
            
            {/* Summary Stats */}
            <div className="animate-in fade-in slide-in-from-bottom-4 rounded-lg border border-foreground/20 bg-foreground/10 p-4 backdrop-blur-md mx-auto max-w-2xl">
              <p className="font-mono text-sm text-foreground/90 mb-2">Inference Summary:</p>
              <p className="text-foreground/80 text-xs">
                <span className="font-semibold">Cumulative Reward:</span> {result.reward.toFixed(6)}
              </p>
              {result.numSteps && (
                <p className="text-foreground/80 text-xs mt-1">
                  <span className="font-semibold">Steps executed:</span> {result.numSteps}
                </p>
              )}
              {result.rewards && result.rewards.length > 1 && (
                <p className="text-foreground/80 text-xs mt-1">
                  <span className="font-semibold">Step rewards:</span> {result.rewards.slice(0, 10).map(r => r.toFixed(6)).join(", ")}
                  {result.rewards.length > 10 && ` ... (${result.rewards.length} total)`}
                </p>
              )}
            </div>
          </div>
        )}
      </div>

    </section>
  )
}
