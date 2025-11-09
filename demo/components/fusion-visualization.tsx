"use client"

import { ChartContainer, ChartTooltip, ChartTooltipContent, ChartLegend, ChartLegendContent } from "@/components/ui/chart"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, AreaChart, Area } from "recharts"

interface FusionObservation {
  profiles?: {
    T_i?: number[]
    T_e?: number[]
    n_e?: number[]
    n_i?: number[]
    psi?: number[]
    v_loop?: number[]
    [key: string]: any
  }
  scalars?: {
    [key: string]: number[]
  }
}

interface FusionVisualizationProps {
  observation: FusionObservation | number[] | null
  action?: {
    ECRH?: number[]
    Ip?: number[]
    NBI?: number[]
    [key: string]: any
  } | null
  reward?: number
  step?: number
}

// Helper to check if observation is structured (has profiles/scalars) or flattened array
function isStructuredObs(obs: any): obs is FusionObservation {
  return obs && typeof obs === 'object' && ('profiles' in obs || 'scalars' in obs)
}

// Helper to create normalized radius array (rho_norm) for plotting
function createRhoNorm(length: number): number[] {
  return Array.from({ length }, (_, i) => i / (length - 1))
}

export function FusionVisualization({ observation, action, reward, step }: FusionVisualizationProps) {
  if (!observation) {
    return (
      <div className="rounded-lg border border-foreground/20 bg-foreground/10 p-6 backdrop-blur-md">
        <p className="text-foreground/60 text-sm">No observation data available</p>
      </div>
    )
  }

  // If observation is flattened array, we can't visualize it meaningfully
  if (Array.isArray(observation)) {
    return (
      <div className="rounded-lg border border-foreground/20 bg-foreground/10 p-6 backdrop-blur-md">
        <p className="text-foreground/60 text-sm mb-2">Observation is flattened ({observation.length} values)</p>
        <p className="text-foreground/40 text-xs">Structured observation data needed for visualization</p>
      </div>
    )
  }

  const profiles = observation.profiles || {}
  const scalars = observation.scalars || {}

  // Prepare temperature profile data
  const tempData = []
  const T_i = profiles.T_i || []
  const T_e = profiles.T_e || []
  const rho_norm = createRhoNorm(Math.max(T_i.length, T_e.length))
  
  for (let i = 0; i < Math.max(T_i.length, T_e.length); i++) {
    tempData.push({
      rho: rho_norm[i] || i / Math.max(T_i.length, T_e.length),
      "T_i (keV)": T_i[i] || 0,
      "T_e (keV)": T_e[i] || 0,
    })
  }

  // Prepare density profile data
  const densityData = []
  const n_e = profiles.n_e || []
  const n_i = profiles.n_i || []
  const rho_norm_density = createRhoNorm(Math.max(n_e.length, n_i.length))
  
  for (let i = 0; i < Math.max(n_e.length, n_i.length); i++) {
    densityData.push({
      rho: rho_norm_density[i] || i / Math.max(n_e.length, n_i.length),
      "n_e (m⁻³)": (n_e[i] || 0) / 1e19, // Convert to 10^19 m^-3
      "n_i (m⁻³)": (n_i[i] || 0) / 1e19,
    })
  }

  // Prepare key scalars
  const keyMetrics = [
    { label: "Thermal Energy (MJ)", value: scalars.W_thermal_total?.[0] ? (scalars.W_thermal_total[0] / 1e6).toFixed(2) : "N/A" },
    { label: "Energy Confinement Time (s)", value: scalars.tau_E?.[0] ? scalars.tau_E[0].toFixed(3) : "N/A" },
    { label: "Q Fusion", value: scalars.Q_fusion?.[0] ? scalars.Q_fusion[0].toFixed(4) : "N/A" },
    { label: "Beta N", value: scalars.beta_N?.[0] ? scalars.beta_N[0].toFixed(3) : "N/A" },
    { label: "q95", value: scalars.q95?.[0] ? scalars.q95[0].toFixed(2) : "N/A" },
    { label: "H98", value: scalars.H98?.[0] ? scalars.H98[0].toFixed(3) : "N/A" },
  ]

  // Prepare action data
  const actionData = []
  if (action) {
    if (action.ECRH && Array.isArray(action.ECRH)) {
      actionData.push({ name: "ECRH Power (W)", value: action.ECRH[0] ? (action.ECRH[0] / 1e6).toFixed(2) + " MW" : "N/A" })
      actionData.push({ name: "ECRH R (m)", value: action.ECRH[1] ? action.ECRH[1].toFixed(3) : "N/A" })
      actionData.push({ name: "ECRH Z (m)", value: action.ECRH[2] ? action.ECRH[2].toFixed(3) : "N/A" })
    }
    if (action.Ip && Array.isArray(action.Ip)) {
      actionData.push({ name: "Plasma Current (A)", value: action.Ip[0] ? (action.Ip[0] / 1e6).toFixed(2) + " MA" : "N/A" })
    }
    if (action.NBI && Array.isArray(action.NBI)) {
      actionData.push({ name: "NBI Power (W)", value: action.NBI[0] ? (action.NBI[0] / 1e6).toFixed(2) + " MW" : "N/A" })
      actionData.push({ name: "NBI R (m)", value: action.NBI[1] ? action.NBI[1].toFixed(3) : "N/A" })
      actionData.push({ name: "NBI Z (m)", value: action.NBI[2] ? action.NBI[2].toFixed(3) : "N/A" })
    }
  }

  const tempConfig = {
    "T_i (keV)": {
      label: "Ion Temperature",
      color: "hsl(var(--chart-1))",
    },
    "T_e (keV)": {
      label: "Electron Temperature",
      color: "hsl(var(--chart-2))",
    },
  }

  const densityConfig = {
    "n_e (m⁻³)": {
      label: "Electron Density",
      color: "hsl(var(--chart-3))",
    },
    "n_i (m⁻³)": {
      label: "Ion Density",
      color: "hsl(var(--chart-4))",
    },
  }

  return (
    <div className="space-y-6 rounded-lg border border-foreground/20 bg-foreground/10 p-6 backdrop-blur-md">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-foreground">Plasma State Visualization</h3>
          {step !== undefined && (
            <p className="text-xs text-foreground/60 mt-1">Step {step}</p>
          )}
        </div>
        {reward !== undefined && (
          <div className="text-right">
            <p className="text-xs text-foreground/60">Reward</p>
            <p className="text-lg font-mono font-semibold text-foreground">{reward.toFixed(6)}</p>
          </div>
        )}
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {keyMetrics.map((metric, idx) => (
          <div key={idx} className="rounded border border-foreground/10 bg-foreground/5 p-3">
            <p className="text-xs text-foreground/60 mb-1">{metric.label}</p>
            <p className="text-sm font-mono font-semibold text-foreground">{metric.value}</p>
          </div>
        ))}
      </div>

      {/* Action Values */}
      {actionData.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-foreground mb-3">Control Actions</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {actionData.map((action, idx) => (
              <div key={idx} className="rounded border border-foreground/10 bg-foreground/5 p-2">
                <p className="text-xs text-foreground/60 mb-1">{action.name}</p>
                <p className="text-xs font-mono font-semibold text-foreground">{action.value}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Temperature Profile */}
      {tempData.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-foreground mb-3">Temperature Profiles</h4>
          <ChartContainer config={tempConfig} className="h-[300px] w-full">
            <LineChart data={tempData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-foreground/10" />
              <XAxis 
                dataKey="rho" 
                label={{ value: "Normalized Radius (ρ)", position: "insideBottom", offset: -5 }}
                className="text-xs"
              />
              <YAxis 
                label={{ value: "Temperature (keV)", angle: -90, position: "insideLeft" }}
                className="text-xs"
              />
              <ChartTooltip content={<ChartTooltipContent />} />
              <ChartLegend content={<ChartLegendContent />} />
              <Line 
                type="monotone" 
                dataKey="T_i (keV)" 
                stroke="var(--color-T_i (keV))" 
                strokeWidth={2}
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="T_e (keV)" 
                stroke="var(--color-T_e (keV))" 
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ChartContainer>
        </div>
      )}

      {/* Density Profile */}
      {densityData.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold text-foreground mb-3">Density Profiles</h4>
          <ChartContainer config={densityConfig} className="h-[300px] w-full">
            <AreaChart data={densityData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-foreground/10" />
              <XAxis 
                dataKey="rho" 
                label={{ value: "Normalized Radius (ρ)", position: "insideBottom", offset: -5 }}
                className="text-xs"
              />
              <YAxis 
                label={{ value: "Density (10¹⁹ m⁻³)", angle: -90, position: "insideLeft" }}
                className="text-xs"
              />
              <ChartTooltip content={<ChartTooltipContent />} />
              <ChartLegend content={<ChartLegendContent />} />
              <Area 
                type="monotone" 
                dataKey="n_e (m⁻³)" 
                stroke="var(--color-n_e (m⁻³))" 
                fill="var(--color-n_e (m⁻³))"
                fillOpacity={0.3}
                strokeWidth={2}
              />
              <Area 
                type="monotone" 
                dataKey="n_i (m⁻³)" 
                stroke="var(--color-n_i (m⁻³))" 
                fill="var(--color-n_i (m⁻³))"
                fillOpacity={0.3}
                strokeWidth={2}
              />
            </AreaChart>
          </ChartContainer>
        </div>
      )}
    </div>
  )
}

