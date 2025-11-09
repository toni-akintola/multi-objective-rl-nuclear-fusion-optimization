"use client"

import { useMemo, useEffect, useState, useRef } from "react"

interface TokamakObservation {
  profiles?: {
    T_i?: number[]
    T_e?: number[]
    n_e?: number[]
    n_i?: number[]
    psi?: number[]
    psi_norm?: number[]
    v_loop?: number[]
    [key: string]: any
  }
  scalars?: {
    R_major?: number[]
    a_minor?: number[]
    [key: string]: any
  }
}

interface TokamakParticlesProps {
  observation: TokamakObservation | null
  action?: {
    ECRH?: number[]
    Ip?: number[]
    NBI?: number[]
    [key: string]: any
  } | null
  step?: number
}

// Helper to check if observation is structured
function isStructuredObs(obs: any): obs is TokamakObservation {
  return obs && typeof obs === 'object' && ('profiles' in obs || 'scalars' in obs)
}

// Create normalized radius array
function createRhoNorm(length: number): number[] {
  return Array.from({ length }, (_, i) => i / Math.max(1, length - 1))
}

// Convert temperature to color
function tempToColor(temp: number, maxTemp: number): string {
  if (maxTemp === 0) return "rgb(100, 150, 255)"
  const normalized = Math.min(1, Math.max(0, temp / maxTemp))
  if (normalized < 0.3) {
    const t = normalized / 0.3
    return `rgb(${100 + t * 100}, ${150 + t * 105}, 255)`
  } else if (normalized < 0.6) {
    const t = (normalized - 0.3) / 0.3
    return `rgb(${200 + t * 55}, ${255}, ${255 - t * 255})`
  } else {
    const t = (normalized - 0.6) / 0.4
    return `rgb(255, ${255 - t * 255}, ${0})`
  }
}

// Particle representation
interface Particle {
  id: number
  rho: number  // Normalized radius (0-1)
  theta: number  // Toroidal angle (0-2π)
  phi: number  // Poloidal angle (0-2π)
  x: number  // Screen x position
  y: number  // Screen y position
  color: string
  size: number
  v_theta: number  // Toroidal velocity
  v_phi: number  // Poloidal velocity
}

export function TokamakParticles({ observation, action, step }: TokamakParticlesProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationFrameRef = useRef<number>()
  const particlesRef = useRef<Particle[]>([])
  const [animationKey, setAnimationKey] = useState(0)

  // Generate particles from observation data
  const particles = useMemo(() => {
    if (!observation || !isStructuredObs(observation)) {
      return []
    }

    const profiles = observation.profiles || {}
    const scalars = observation.scalars || {}
    
    const R_major = scalars.R_major?.[0] || 6.2
    const a_minor = scalars.a_minor?.[0] || 2.0
    
    const T_i = profiles.T_i || []
    const T_e = profiles.T_e || []
    const n_e = profiles.n_e || []
    const v_loop = profiles.v_loop || []
    const psi_norm = profiles.psi_norm || createRhoNorm(Math.max(T_i.length, T_e.length, n_e.length))
    
    const maxTemp = Math.max(
      ...T_i.map(t => t || 0),
      ...T_e.map(t => t || 0),
      1
    )
    const maxDensity = Math.max(...n_e.map(n => n || 0), 1)
    
    // Generate particles based on density distribution
    const numParticles = 500  // Total number of particles to show
    const newParticles: Particle[] = []
    
    for (let i = 0; i < numParticles; i++) {
      // Sample rho based on density distribution (more particles where density is higher)
      const rand = Math.random()
      let rho = 0
      let cumulativeDensity = 0
      const totalDensity = n_e.reduce((sum, n) => sum + (n || 0), 0)
      
      if (totalDensity > 0) {
        for (let j = 0; j < n_e.length; j++) {
          const rho_j = psi_norm[j] || (j / Math.max(1, n_e.length - 1))
          cumulativeDensity += (n_e[j] || 0) / totalDensity
          if (cumulativeDensity >= rand) {
            rho = rho_j
            break
          }
        }
      } else {
        rho = Math.random()
      }
      
      // Random toroidal and poloidal angles
      const theta = Math.random() * Math.PI * 2  // Toroidal angle
      const phi = Math.random() * Math.PI * 2  // Poloidal angle
      
      // Get temperature at this radius for coloring
      const rhoIdx = Math.min(Math.floor(rho * (T_i.length - 1)), T_i.length - 1)
      const temp = (T_i[rhoIdx] || 0 + T_e[rhoIdx] || 0) / 2
      const color = tempToColor(temp, maxTemp)
      
      // Get density for particle size
      const density = n_e[rhoIdx] || 0
      const size = 1 + (density / maxDensity) * 3  // Size between 1-4 pixels
      
      // Get velocity for animation
      const v = v_loop[rhoIdx] || 0
      const v_theta = v * 0.1  // Toroidal velocity component
      const v_phi = v * 0.05  // Poloidal velocity component
      
      // Calculate 2D projection position (poloidal cross-section)
      const r = a_minor * rho
      const x_local = r * Math.cos(phi)
      const y_local = r * Math.sin(phi)
      
      // Project to 2D view (looking down the torus axis)
      const x = R_major + x_local
      const y = y_local
      
      newParticles.push({
        id: i,
        rho,
        theta,
        phi,
        x,
        y,
        color,
        size,
        v_theta,
        v_phi,
      })
    }
    
    return newParticles
  }, [observation, step])

  // Update particles reference when observation changes
  useEffect(() => {
    particlesRef.current = particles
    setAnimationKey(prev => prev + 1)
  }, [particles])

  // Animation loop
  useEffect(() => {
    if (!canvasRef.current || particles.length === 0) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * window.devicePixelRatio
    canvas.height = rect.height * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    let animationTime = 0

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width / window.devicePixelRatio, canvas.height / window.devicePixelRatio)
      
      // Draw background
      ctx.fillStyle = 'rgba(0, 0, 0, 0.9)'
      ctx.fillRect(0, 0, canvas.width / window.devicePixelRatio, canvas.height / window.devicePixelRatio)

      // Get current particles
      const currentParticles = particlesRef.current
      if (currentParticles.length === 0) {
        animationFrameRef.current = requestAnimationFrame(animate)
        return
      }

      // Calculate view transform
      const R_major = 6.2
      const a_minor = 2.0
      const viewWidth = (R_major + a_minor) * 2
      const viewHeight = a_minor * 2.5
      const viewX = R_major - a_minor - 1
      const viewY = -a_minor - 0.5

      const scaleX = (canvas.width / window.devicePixelRatio) / viewWidth
      const scaleY = (canvas.height / window.devicePixelRatio) / viewHeight
      const offsetX = -viewX * scaleX
      const offsetY = -viewY * scaleY

      // Draw particles
      animationTime += 0.016  // ~60fps
      
      currentParticles.forEach(particle => {
        // Update angles based on velocity
        const newTheta = particle.theta + particle.v_theta * animationTime
        const newPhi = particle.phi + particle.v_phi * animationTime
        
        // Calculate position
        const r = a_minor * particle.rho
        const x_local = r * Math.cos(newPhi)
        const y_local = r * Math.sin(newPhi)
        const x = R_major + x_local
        const y = y_local
        
        // Transform to screen coordinates
        const screenX = x * scaleX + offsetX
        const screenY = y * scaleY + offsetY
        
        // Draw particle
        ctx.beginPath()
        ctx.arc(screenX, screenY, particle.size, 0, Math.PI * 2)
        ctx.fillStyle = particle.color
        ctx.globalAlpha = 0.7
        ctx.fill()
        
        // Add glow effect
        const gradient = ctx.createRadialGradient(screenX, screenY, 0, screenX, screenY, particle.size * 2)
        gradient.addColorStop(0, particle.color)
        gradient.addColorStop(1, 'transparent')
        ctx.fillStyle = gradient
        ctx.globalAlpha = 0.3
        ctx.fill()
        ctx.globalAlpha = 1.0
      })

      animationFrameRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [animationKey, particles.length])

  if (!observation || !isStructuredObs(observation)) {
    return (
      <div className="rounded-lg border border-foreground/20 bg-foreground/10 p-6 backdrop-blur-md flex items-center justify-center h-[600px]">
        <p className="text-foreground/60 text-sm">No observation data available for particle visualization</p>
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-foreground/20 bg-foreground/10 backdrop-blur-md overflow-hidden w-full">
      <div className="p-4 bg-background/30">
        <p className="font-mono text-sm text-foreground/90 mb-2">
          Particle Dynamics Visualization {step !== undefined ? `(Step: ${step})` : ''}
        </p>
        <p className="text-xs text-foreground/60">
          Showing ~500 particles sampled from density distribution, colored by temperature, animated by velocity
        </p>
      </div>
      <div className="bg-black/50 p-4">
        <canvas
          ref={canvasRef}
          className="w-full h-[500px]"
          style={{ imageRendering: 'pixelated' }}
        />
      </div>
    </div>
  )
}

