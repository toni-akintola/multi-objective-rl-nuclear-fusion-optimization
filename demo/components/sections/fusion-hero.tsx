"use client"

import { MagneticButton } from "@/components/magnetic-button"
import { Tokamak3DR3F } from "@/components/tokamak-3d-r3f"
import { useState } from "react"
import { ResponsiveLine } from "@nivo/line"

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
  const [randomAgentRewards, setRandomAgentRewards] = useState<number[]>([])
  const [pidAgentRewards, setPidAgentRewards] = useState<number[]>([])

  const handleRunInference = async () => {
    console.log("🔵 [FRONTEND] Run Inference button clicked")
    setIsRunning(true)
    setResult(null)
    setCurrentStep(0)
    setRandomAgentRewards([])
    setPidAgentRewards([])
    
    const startTime = Date.now()
    const numSteps = 1000  // Can be increased further if needed
    
    try {
      // Reset all environments in parallel
      console.log("📡 [FRONTEND] Resetting environments...")
      const [resetResponse, randomResetResponse, pidResetResponse] = await Promise.all([
        fetch("http://localhost:8000/reset", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        }),
        fetch("http://localhost:8000/random_reset", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        }),
        fetch("http://localhost:8000/pid_reset", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        }).catch(err => {
          console.warn("PID agent reset failed, continuing without PID agent:", err)
          return null
        })
      ])
      
      if (!resetResponse.ok) {
        throw new Error(`Reset failed: ${resetResponse.status}`)
      }
      if (!randomResetResponse.ok) {
        console.warn("Random agent reset failed, continuing without random agent comparison")
      }
      if (pidResetResponse && !pidResetResponse.ok) {
        console.warn("PID agent reset failed, continuing without PID agent comparison")
      }
      
      const resetData = await resetResponse.json()
      console.log("✅ [FRONTEND] Environments reset")
      
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
      let randomCumulativeReward = 0
      const randomRewards: number[] = []
      let pidCumulativeReward = 0
      const pidRewards: number[] = []
      
      for (let step = 0; step < numSteps; step++) {
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), 5000) // 5 second timeout per step
        
        try {
          // Run all agents in parallel
          const [response, randomResponse, pidResponse] = await Promise.all([
            fetch(`http://localhost:8000/step?deterministic=true&num_steps=1`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              signal: controller.signal,
            }),
            fetch("http://localhost:8000/random_step", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              signal: controller.signal,
            }).catch(err => {
              // If random agent fails, continue without it
              console.warn("Random agent step failed:", err)
              return null
            }),
            fetch("http://localhost:8000/pid_step", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              signal: controller.signal,
            }).catch(err => {
              // If PID agent fails, continue without it
              console.warn("PID agent step failed:", err)
              return null
            })
          ])
          
          clearTimeout(timeoutId)
          
          if (!response.ok) {
            console.error(`❌ [FRONTEND] HTTP error! status: ${response.status}`)
            throw new Error(`HTTP error! status: ${response.status}`)
          }
          
          const data = await response.json()
          cumulativeReward += data.reward
          rewards.push(data.reward)
          
          // Process random agent response if available
          if (randomResponse && randomResponse.ok) {
            const randomData = await randomResponse.json()
            randomCumulativeReward += randomData.reward
            randomRewards.push(randomData.reward)
            setRandomAgentRewards([...randomRewards])
          }
          
          // Process PID agent response if available
          if (pidResponse && pidResponse.ok) {
            const pidData = await pidResponse.json()
            pidCumulativeReward += pidData.reward
            pidRewards.push(pidData.reward)
            setPidAgentRewards([...pidRewards])
          }
          
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

  return (
    <section className="flex min-h-screen w-screen shrink-0 flex-col items-center justify-start px-6 pt-24 pb-24 md:px-12 md:pt-32 md:pb-32 lg:pt-36 lg:pb-36">
      <div className="max-w-3xl w-full text-center">
        <h1 className="mb-6 animate-in fade-in slide-in-from-bottom-8 font-sans text-6xl font-light leading-[1.1] tracking-tight text-foreground duration-1000 md:text-7xl lg:text-8xl">
          <span className="text-balance">
            Controlling
            <br />
            limitless energy.
          </span>
        </h1>
        <p className="mb-8 mx-auto max-w-xl animate-in fade-in slide-in-from-bottom-4 text-lg leading-relaxed text-foreground/80 duration-1000 delay-200 md:text-xl">
          <span className="text-pretty">
            Harnessing reinforcement learning to master plasma dynamics and unlock the potential of fusion energy.
          </span>
        </p>
        <div className="flex animate-in fade-in slide-in-from-bottom-4 flex-col gap-4 duration-1000 delay-300 sm:flex-row sm:items-center sm:justify-center">
          <MagneticButton 
            size="lg" 
            variant="primary" 
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
            
            {/* Live Reward Chart */}
            {result.rewards && result.rewards.length > 0 && (
              <div className="animate-in fade-in slide-in-from-bottom-4 rounded-lg border border-foreground/20 bg-foreground/10 p-4 backdrop-blur-md mx-auto max-w-4xl h-64">
                <p className="font-mono text-sm text-foreground/90 mb-4">Cumulative Reward Over Time</p>
                <ResponsiveLine
                  data={[
                    {
                      id: "trained_agent",
                      data: result.rewards.map((reward, index) => {
                        const cumulative = result.rewards!.slice(0, index + 1).reduce((sum, r) => sum + r, 0)
                        return {
                          x: index + 1,
                          y: cumulative,
                        }
                      }),
                    },
                    ...(randomAgentRewards.length > 0 ? [{
                      id: "random_agent",
                      data: randomAgentRewards.map((reward, index) => {
                        const cumulative = randomAgentRewards.slice(0, index + 1).reduce((sum, r) => sum + r, 0)
                        return {
                          x: index + 1,
                          y: cumulative,
                        }
                      }),
                    }] : []),
                    ...(pidAgentRewards.length > 0 ? [{
                      id: "pid_agent",
                      data: pidAgentRewards.map((reward, index) => {
                        const cumulative = pidAgentRewards.slice(0, index + 1).reduce((sum, r) => sum + r, 0)
                        return {
                          x: index + 1,
                          y: cumulative,
                        }
                      }),
                    }] : []),
                  ]}
                  margin={{ top: 20, right: 120, bottom: 50, left: 60 }}
                  xScale={{ type: "linear", min: 0 }}
                  yScale={{ type: "linear", min: "auto" }}
                  curve="monotoneX"
                  axisTop={null}
                  axisRight={null}
                  axisBottom={{
                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: 0,
                    legend: "Step",
                    legendPosition: "middle",
                    legendOffset: 40,
                  }}
                  axisLeft={{
                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: 0,
                    legend: "Cumulative Reward",
                    legendPosition: "middle",
                    legendOffset: -50,
                  }}
                  pointSize={0}
                  pointColor={{ theme: "background" }}
                  pointBorderWidth={0}
                  pointBorderColor={{ from: "serieColor" }}
                  enableArea={false}
                  useMesh={true}
                  colors={(d) => {
                    if (d.id === "random_agent") {
                      return "#ef4444" // red-500
                    }
                    if (d.id === "pid_agent") {
                      return "#10b981" // green-500
                    }
                    return "#3b82f6" // blue-500 (default - trained agent)
                  }}
                  lineWidth={2}
                  legends={[
                    {
                      anchor: "right",
                      direction: "column",
                      justify: false,
                      translateX: 120,
                      translateY: 0,
                      itemsSpacing: 8,
                      itemDirection: "left-to-right",
                      itemWidth: 100,
                      itemHeight: 20,
                      symbolSize: 12,
                      symbolShape: "circle",
                    },
                  ]}
                  theme={{
                    text: {
                      fill: "rgba(255, 255, 255, 0.7)",
                      fontSize: 11,
                    },
                    axis: {
                      domain: {
                        line: {
                          stroke: "rgba(255, 255, 255, 0.2)",
                          strokeWidth: 1,
                        },
                      },
                      ticks: {
                        line: {
                          stroke: "rgba(255, 255, 255, 0.2)",
                          strokeWidth: 1,
                        },
                        text: {
                          fill: "rgba(255, 255, 255, 0.7)",
                        },
                      },
                    },
                    grid: {
                      line: {
                        stroke: "rgba(255, 255, 255, 0.1)",
                        strokeWidth: 1,
                      },
                    },
                  }}
                />
              </div>
            )}
          </div>
        )}
      </div>

    </section>
  )
}
