"use client"

import { useMemo, useRef, useEffect } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import { OrbitControls, PerspectiveCamera, Text } from "@react-three/drei"
import * as THREE from "three"

interface TokamakObservation {
  profiles?: {
    T_i?: number[]
    T_e?: number[]
    n_e?: number[]
    n_i?: number[]
    psi?: number[]
    psi_norm?: number[]
    v_loop?: number[]
    elongation?: number[]
    [key: string]: any
  }
  scalars?: {
    R_major?: number[]
    a_minor?: number[]
    [key: string]: any
  }
}

interface Tokamak3DR3FProps {
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
function tempToColor(temp: number, maxTemp: number): THREE.Color {
  if (maxTemp === 0) return new THREE.Color(0.4, 0.6, 1.0)
  const normalized = Math.min(1, Math.max(0, temp / maxTemp))
  if (normalized < 0.3) {
    const t = normalized / 0.3
    return new THREE.Color(0.4 + t * 0.4, 0.6 + t * 0.4, 1.0)
  } else if (normalized < 0.6) {
    const t = (normalized - 0.3) / 0.3
    return new THREE.Color(0.8 + t * 0.2, 1.0, 1.0 - t)
  } else {
    const t = (normalized - 0.6) / 0.4
    return new THREE.Color(1.0, 1.0 - t, 0)
  }
}

// Plasma Torus Component
function PlasmaTorus({ observation }: { observation: TokamakObservation }) {
  const meshRef = useRef<THREE.Mesh>(null)
  const materialRef = useRef<THREE.MeshStandardMaterial>(null)
  
  const { geometry, maxTemp, maxDensity } = useMemo(() => {
    if (!observation || !isStructuredObs(observation)) {
      return { geometry: null, maxTemp: 1, maxDensity: 1 }
    }

    const profiles = observation.profiles || {}
    const scalars = observation.scalars || {}
    
    const R_major = scalars.R_major?.[0] || 6.2
    const a_minor = scalars.a_minor?.[0] || 2.0
    const elongation = profiles.elongation?.[0] || 1.5
    
    const T_i = profiles.T_i || []
    const T_e = profiles.T_e || []
    const n_e = profiles.n_e || []
    const psi_norm = profiles.psi_norm || createRhoNorm(Math.max(T_i.length, T_e.length, n_e.length))
    
    const maxTemp = Math.max(
      ...T_i.map(t => t || 0),
      ...T_e.map(t => t || 0),
      1
    )
    const maxDensity = Math.max(...n_e.map(n => n || 0), 1)
    
    // Create torus geometry
    const segments = 64
    const tubeSegments = 32
    const geometry = new THREE.TorusGeometry(R_major, a_minor, segments, tubeSegments)
    
    // Modify vertices for elongation
    const positions = geometry.attributes.position.array as Float32Array
    const colors = new Float32Array(positions.length)
    
    for (let i = 0; i < positions.length; i += 3) {
      const x = positions[i]
      const y = positions[i + 1]
      const z = positions[i + 2]
      
      // Calculate distance from torus center
      const r = Math.sqrt(x * x + z * z)
      const angle = Math.atan2(z, x)
      
      // Apply elongation to y-axis
      positions[i + 1] = y * elongation
      
      // Calculate normalized radius for color mapping
      const normalizedR = Math.min(1, Math.abs((r - R_major) / a_minor))
      const rhoIdx = Math.min(Math.floor(normalizedR * (T_i.length - 1)), T_i.length - 1)
      const temp = (T_i[rhoIdx] || 0 + T_e[rhoIdx] || 0) / 2
      const density = n_e[rhoIdx] || 0
      
      // Set color based on temperature
      const color = tempToColor(temp, maxTemp)
      colors[i] = color.r
      colors[i + 1] = color.g
      colors[i + 2] = color.b
    }
    
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geometry.computeVertexNormals()
    
    return { geometry, maxTemp, maxDensity }
  }, [observation])
  
  useFrame((state) => {
    if (meshRef.current) {
      // Rotate around vertical axis (z-axis when horizontal)
      meshRef.current.rotation.z += 0.001
    }
  })
  
  if (!geometry) {
    return null
  }
  
  return (
    <mesh ref={meshRef} geometry={geometry} rotation={[Math.PI / 2, 0, 0]}>
      <meshStandardMaterial
        vertexColors
        emissive={new THREE.Color(1, 0.5, 0.2)}
        emissiveIntensity={0.5}
        transparent
        opacity={0.8}
        metalness={0.1}
        roughness={0.3}
      />
    </mesh>
  )
}

// Particle System Component
function ParticleSystem({ observation }: { observation: TokamakObservation }) {
  const particlesRef = useRef<THREE.Points>(null)
  
  const { positions, colors, sizes } = useMemo(() => {
    if (!observation || !isStructuredObs(observation)) {
      return { positions: new Float32Array(0), colors: new Float32Array(0), sizes: new Float32Array(0) }
    }

    const profiles = observation.profiles || {}
    const scalars = observation.scalars || {}
    
    const R_major = scalars.R_major?.[0] || 6.2
    const a_minor = scalars.a_minor?.[0] || 2.0
    const elongation = profiles.elongation?.[0] || 1.5
    
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
    const numParticles = 1000
    const positions = new Float32Array(numParticles * 3)
    const colors = new Float32Array(numParticles * 3)
    const sizes = new Float32Array(numParticles)
    
    for (let i = 0; i < numParticles; i++) {
      // Sample rho based on density distribution (more particles where density is higher)
      const rand = Math.random()
      let rho = 0
      let cumulativeDensity = 0
      const totalDensity = n_e.reduce((sum, n) => sum + (n || 0), 0)
      
      if (totalDensity > 0) {
        for (let j = 0; j < n_e.length; j++) {
          cumulativeDensity += (n_e[j] || 0) / totalDensity
          if (cumulativeDensity >= rand) {
            rho = psi_norm[j] || (j / Math.max(1, n_e.length - 1))
            break
          }
        }
      } else {
        rho = Math.random()
      }
      
      // Random toroidal and poloidal angles
      const theta = Math.random() * Math.PI * 2  // Toroidal angle
      const phi = Math.random() * Math.PI * 2  // Poloidal angle
      
      // Calculate position in 3D space (matching horizontal torus orientation)
      const r = a_minor * rho
      const x_local = r * Math.cos(phi)
      const y_local = r * Math.sin(phi) * elongation
      const z_local = 0
      
      // Transform to toroidal coordinates (horizontal orientation)
      const R = R_major + x_local
      const x = R * Math.cos(theta)
      const y = R * Math.sin(theta)  // Swapped for horizontal
      const z = y_local  // Swapped for horizontal
      
      positions[i * 3] = x
      positions[i * 3 + 1] = y
      positions[i * 3 + 2] = z
      
      // Color based on temperature
      const rhoIdx = Math.min(Math.floor(rho * (T_i.length - 1)), T_i.length - 1)
      const temp = ((T_i[rhoIdx] || 0) + (T_e[rhoIdx] || 0)) / 2
      const color = tempToColor(temp, maxTemp)
      colors[i * 3] = color.r
      colors[i * 3 + 1] = color.g
      colors[i * 3 + 2] = color.b
      
      // Size based on density
      const density = n_e[rhoIdx] || 0
      sizes[i] = 0.1 + (density / maxDensity) * 0.3
    }
    
    return { positions, colors, sizes }
  }, [observation])
  
  // Create and update geometry
  const geometry = useMemo(() => {
    if (positions.length === 0) return null
    
    const geom = new THREE.BufferGeometry()
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geom.setAttribute('size', new THREE.BufferAttribute(sizes, 1))
    return geom
  }, [positions, colors, sizes])
  
  useFrame((state, delta) => {
    if (particlesRef.current) {
      // Rotate particles around torus (z-axis for horizontal orientation)
      particlesRef.current.rotation.z += 0.01 * delta
    }
  })
  
  if (!geometry || positions.length === 0) {
    return null
  }
  
  return (
    <points ref={particlesRef} geometry={geometry} rotation={[Math.PI / 2, 0, 0]}>
      <pointsMaterial
        vertexColors
        size={0.3}
        sizeAttenuation={true}
        transparent
        opacity={0.9}
      />
    </points>
  )
}

// Vessel Component
function TokamakVessel({ observation }: { observation: TokamakObservation | null }) {
  const vesselGeometry = useMemo(() => {
    const R_major = observation?.scalars?.R_major?.[0] || 6.2
    const a_minor = observation?.scalars?.a_minor?.[0] || 2.0
    const elongation = observation?.profiles?.elongation?.[0] || 1.5
    
    const vesselRadius = a_minor * 1.2
    const geometry = new THREE.TorusGeometry(R_major, vesselRadius, 64, 32)
    
    const positions = geometry.attributes.position.array as Float32Array
    for (let i = 0; i < positions.length; i += 3) {
      positions[i + 1] *= elongation
    }
    
    geometry.computeVertexNormals()
    return geometry
  }, [observation])
  
  return (
    <mesh geometry={vesselGeometry} rotation={[Math.PI / 2, 0, 0]}>
      <meshStandardMaterial
        color="#2a2a2a"
        metalness={0.8}
        roughness={0.2}
        transparent
        opacity={0.4}
      />
    </mesh>
  )
}

// Magnetic Coils
function MagneticCoils({ observation }: { observation: TokamakObservation | null }) {
  const R_major = observation?.scalars?.R_major?.[0] || 6.2
  const numCoils = 16
  
  return (
    <>
      {Array.from({ length: numCoils }).map((_, i) => {
        const angle = (i / numCoils) * Math.PI * 2
        const coilRadius = 0.3
        const coilPosition = R_major * 1.5
        
        return (
          <mesh
            key={i}
            position={[
              Math.cos(angle) * coilPosition,
              0,
              Math.sin(angle) * coilPosition
            ]}
            rotation={[0, angle, Math.PI / 2]}
          >
            <torusGeometry args={[coilRadius, 0.05, 16, 16]} />
            <meshStandardMaterial
              color="#4a90e2"
              metalness={0.9}
              roughness={0.1}
              emissive="#1a4a7a"
              emissiveIntensity={0.3}
            />
          </mesh>
        )
      })}
    </>
  )
}

// Control Actuators
function ControlActuators({ action, observation }: { action?: any, observation: TokamakObservation | null }) {
  const R_major = observation?.scalars?.R_major?.[0] || 6.2
  const a_minor = observation?.scalars?.a_minor?.[0] || 2.0
  
  return (
    <>
      {/* ECRH heating */}
      {action?.ECRH && Array.isArray(action.ECRH) && action.ECRH[0] > 0 && (
        <mesh position={[R_major + a_minor * 1.1, 0, 0]}>
          <coneGeometry args={[0.2, 1, 8]} />
          <meshStandardMaterial
            color="#ff6b6b"
            emissive="#ff0000"
            emissiveIntensity={0.8}
          />
        </mesh>
      )}
      
      {/* NBI heating */}
      {action?.NBI && Array.isArray(action.NBI) && action.NBI[0] > 0 && (
        <mesh position={[R_major + a_minor * 1.1, a_minor * 0.5, 0]}>
          <coneGeometry args={[0.2, 1, 8]} />
          <meshStandardMaterial
            color="#4ecdc4"
            emissive="#00ffff"
            emissiveIntensity={0.8}
          />
        </mesh>
      )}
    </>
  )
}

// Main Scene Component
function TokamakScene({ observation, action }: Tokamak3DR3FProps) {
  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />
      <directionalLight position={[0, 10, 0]} intensity={0.8} />
      
      {/* Tokamak components - plasma torus and particles */}
      {observation && isStructuredObs(observation) && (
        <>
          <PlasmaTorus observation={observation} />
          <ParticleSystem observation={observation} />
        </>
      )}
    </>
  )
}

export function Tokamak3DR3F({ observation, action, step }: Tokamak3DR3FProps) {
  if (!observation || !isStructuredObs(observation)) {
    return (
      <div className="rounded-lg border border-foreground/20 bg-foreground/10 p-6 backdrop-blur-md flex items-center justify-center h-[600px]">
        <p className="text-foreground/60 text-sm">No observation data available for 3D visualization</p>
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-foreground/20 bg-foreground/10 backdrop-blur-md overflow-hidden w-full">
      <div className="p-4 bg-background/30">
        <p className="font-mono text-sm text-foreground/90 mb-2">
          3D Tokamak Visualization {step !== undefined ? `(Step: ${step})` : ''}
        </p>
        <p className="text-xs text-foreground/60">
          Interactive 3D model with plasma torus and particle system showing atomic movement
        </p>
      </div>
      <div className="bg-black/50 h-[600px] w-full">
        <Canvas shadows>
          <PerspectiveCamera makeDefault position={[0, 15, 0]} fov={50} />
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            minDistance={8}
            maxDistance={40}
          />
          <TokamakScene observation={observation} action={action} />
        </Canvas>
      </div>
    </div>
  )
}

