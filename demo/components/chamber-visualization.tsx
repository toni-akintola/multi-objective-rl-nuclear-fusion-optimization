"use client"

import { useEffect, useRef, useState } from "react"

interface ChamberData {
  beta_N: number
  q_min: number
  q95: number
  ok: boolean
  violation: number
  in_box: boolean
  smooth: boolean
  is_corrective?: boolean
}

export function ChamberVisualization() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [connected, setConnected] = useState(false)
  const [data, setData] = useState<ChamberData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const animationFrameRef = useRef<number | undefined>(undefined)
  const trailsRef = useRef<Array<{
    angle: number
    radius: number
    speed: number
    phase: number
    color: string
  }>>([])
  const timeRef = useRef(0)
  const violationHistoryRef = useRef<number[]>([])
  const prevValuesRef = useRef<{ beta_N?: number; q_min?: number; q95?: number }>({} as { beta_N?: number; q_min?: number; q95?: number })

  // Initialize particle trails (same as visualize_chamber_live.py)
  useEffect(() => {
    const numTrails = 300
    trailsRef.current = []
    for (let i = 0; i < numTrails; i++) {
      const angle = Math.random() * 2 * Math.PI
      const radius = 3 + Math.random() * 4
      trailsRef.current.push({
        angle,
        radius,
        speed: 0.05 + Math.random() * 0.1,
        phase: Math.random() * 2 * Math.PI,
        color: "cyan",
      })
    }
  }, [])

  // Track violation history
  useEffect(() => {
    if (data) {
      violationHistoryRef.current.push(data.violation)
      if (violationHistoryRef.current.length > 50) {
        violationHistoryRef.current.shift()
      }
    }
  }, [data])

  // WebSocket connection
  useEffect(() => {
    let ws: WebSocket | null = null
    let reconnectTimeout: NodeJS.Timeout | null = null
    let reconnectAttempts = 0
    const maxReconnectAttempts = 5

    const connect = () => {
      try {
        ws = new WebSocket("ws://localhost:8765")

        ws.onopen = () => {
          setConnected(true)
          setError(null)
          reconnectAttempts = 0
          console.log("✅ Connected to visualization server")
        }

        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data)
            if (message.type === "chamber_data") {
              setData(message.data)
            }
          } catch (err) {
            console.error("Error parsing message:", err)
          }
        }

        ws.onerror = (err) => {
          console.error("WebSocket error:", err)
        }

        ws.onclose = (event) => {
          setConnected(false)
          console.log("Disconnected from visualization server", event.code, event.reason)
          
          if (event.code !== 1000 && reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000)
            console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})...`)
            reconnectTimeout = setTimeout(connect, delay)
          } else if (reconnectAttempts >= maxReconnectAttempts) {
            setError("Failed to connect after multiple attempts. Make sure the Python server is running: python chamber_websocket_server.py")
          } else {
            setError("Connection closed. Make sure the Python server is running.")
          }
        }
      } catch (err) {
        console.error("Failed to create WebSocket:", err)
        setError("Failed to create WebSocket connection. Check browser console for details.")
      }
    }

    connect()

    return () => {
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout)
      }
      if (ws) {
        ws.close()
      }
    }
  }, [])

  // Animation loop - matches visualize_chamber_live.py exactly
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const draw = () => {
      const width = canvas.width
      const height = canvas.height
      
      // Match Python: xlim(-12, 12), ylim(-12, 8)
      // Python coordinate system: x from -12 to 12, y from -12 to 8
      // World center is at (0, 0) but viewport center in y is at (-12+8)/2 = -2
      const xMin = -12
      const xMax = 12
      const yMin = -12
      const yMax = 8
      const xRange = xMax - xMin  // 24
      const yRange = yMax - yMin   // 20
      
      // Calculate scale to fit both dimensions
      const scaleX = width / xRange
      const scaleY = height / yRange
      const scale = Math.min(scaleX, scaleY)  // Use smaller to maintain aspect ratio
      
      // Calculate center offsets
      const centerX = width / 2
      const centerY = height / 2
      
      // Convert world coordinates to canvas coordinates
      const worldToCanvasX = (worldX: number) => centerX + worldX * scale
      const worldToCanvasY = (worldY: number) => centerY - worldY * scale  // Flip Y axis

      // Clear canvas - same color as Python
      ctx.fillStyle = "#0a0a0f"
      ctx.fillRect(0, 0, width, height)

      if (!data) {
        animationFrameRef.current = requestAnimationFrame(draw)
        return
      }

      // Chamber walls (outermost) - radius 10 at (0, 0)
      ctx.strokeStyle = "#3a3a4a"
      ctx.lineWidth = 3 * scale
      ctx.globalAlpha = 0.6
      ctx.beginPath()
      ctx.arc(worldToCanvasX(0), worldToCanvasY(0), 10 * scale, 0, 2 * Math.PI)
      ctx.stroke()

      // Inner wall - radius 9 at (0, 0)
      ctx.strokeStyle = "#2a2a3a"
      ctx.lineWidth = 1 * scale
      ctx.globalAlpha = 0.4
      ctx.setLineDash([5 * scale, 5 * scale])
      ctx.beginPath()
      ctx.arc(worldToCanvasX(0), worldToCanvasY(0), 9 * scale, 0, 2 * Math.PI)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.globalAlpha = 1.0

      // Center column - animated, at (0, 0)
      const colRadius = 1.5 * scale * (1 + 0.2 * Math.sin(timeRef.current))
      ctx.fillStyle = "#1a1a2a"
      ctx.strokeStyle = "#4a4a5a"
      ctx.lineWidth = 2 * scale
      ctx.globalAlpha = 0.8
      ctx.beginPath()
      ctx.arc(worldToCanvasX(0), worldToCanvasY(0), colRadius, 0, 2 * Math.PI)
      ctx.fill()
      ctx.stroke()
      ctx.globalAlpha = 1.0

      // Update trails and colors (same logic as Python)
      const sizeFactor = 0.7 + ((data.beta_N - 0.5) / (3.0 - 0.5)) * 0.3
      const plasmaBoundaryRadius = 3 + ((data.beta_N - 0.5) / (3.0 - 0.5)) * 4

      // Update colors based on status (weighted random like Python)
      const getColorForStatus = (ok: boolean, isCorrective: boolean, violation: number): string => {
        const r = Math.random()
        if (ok) {
          // p=[0.4, 0.4, 0.2] for cyan, blue, green
          if (r < 0.4) return "cyan"
          if (r < 0.8) return "blue"
          return "green"
        } else if (isCorrective) {
          // p=[0.5, 0.3, 0.2] for orange, yellow, lime
          if (r < 0.5) return "orange"
          if (r < 0.8) return "yellow"
          return "lime"
        } else if (violation < 0.5) {
          // p=[0.7, 0.3] for orange, yellow
          return r < 0.7 ? "orange" : "yellow"
        } else {
          // p=[0.5, 0.3, 0.2] for red, magenta, pink
          if (r < 0.5) return "red"
          if (r < 0.8) return "magenta"
          return "pink"
        }
      }

      trailsRef.current.forEach((trail) => {
        trail.angle += trail.speed * sizeFactor
        trail.phase += 0.02
        
        // Update color every frame (exactly like Python)
        trail.color = getColorForStatus(data.ok, data.is_corrective || false, data.violation)

        // Calculate radius (keep within plasma boundary)
        let radius = trail.radius * sizeFactor
        if (radius > plasmaBoundaryRadius * 0.95) {
          radius = plasmaBoundaryRadius * 0.95
        }

        const x = worldToCanvasX(radius * Math.cos(trail.angle))
        const y = worldToCanvasY(radius * Math.sin(trail.angle))

        // Draw trail streak (5 points exactly like Python)
        for (let i = 0; i < 5; i++) {
          const t = i / 5
          const prevAngle = trail.angle - trail.speed * (1 - t) * 10
          const prevRadius = radius * (1 - t * 0.1)
          const px = worldToCanvasX(prevRadius * Math.cos(prevAngle))
          const py = worldToCanvasY(prevRadius * Math.sin(prevAngle))
          
          const alpha = (1 - t) * 0.8  // Exact match: Python uses 0.8
          const size = (20 * (1 - t) + 5) / 10  // Match Python scatter size
          
          ctx.fillStyle = trail.color
          ctx.globalAlpha = alpha
          ctx.beginPath()
          ctx.arc(px, py, size, 0, 2 * Math.PI)
          ctx.fill()
        }

        // Main particle (exactly like Python: s=30, alpha=0.9, linewidths=0.5)
        ctx.fillStyle = trail.color
        ctx.strokeStyle = "white"
        ctx.lineWidth = 0.5  // Match Python linewidths=0.5
        ctx.globalAlpha = 0.9  // Match Python alpha=0.9
        ctx.beginPath()
        ctx.arc(x, y, 1.5, 0, 2 * Math.PI)  // Match Python s=30 (radius ~1.5)
        ctx.fill()
        ctx.stroke()
        ctx.globalAlpha = 1.0
      })

      // Status (exactly like Python)
      let status: string
      let statusColor: string
      if (data.ok) {
        status = "🟢 SAFE"
        statusColor = "#00ff00"
      } else if (data.is_corrective) {
        status = "🟠 SELF-FIXING! (Severity ↓)"
        statusColor = "#ff8800"
      } else {
        status = "🔴 VIOLATION"
        statusColor = "#ff0000"
      }

      // Draw plasma boundary (D-shaped, elongated) - exact Python formula
      // Python: base_radius = 3 + (beta_N - 0.5) / (3.0 - 0.5) * 4
      const baseRadius = 3 + ((data.beta_N - 0.5) / (3.0 - 0.5)) * 4
      const elongation = 1.0 + ((data.q95 - 3.0) / (5.0 - 3.0)) * 0.3
      const triangularity = 0.0 + ((1.0 - data.q_min) / (1.0 - 0.5)) * 0.2
      
      // Python: R_plasma = R0 + a * (cos(theta) + triangularity * cos(2*theta))
      // Z_plasma = a * elongation * sin(theta)
      // Then plots (R_plasma, Z_plasma) as (x, y) coordinates
      const R0 = 0.0
      const a = baseRadius

      ctx.strokeStyle = statusColor
      ctx.lineWidth = 3 * scale
      ctx.globalAlpha = 0.9
      ctx.beginPath()

      // Use exactly 100 points like Python: np.linspace(0, 2*pi, 100)
      for (let i = 0; i <= 100; i++) {
        const theta = (i / 100) * 2 * Math.PI
        const R_plasma = R0 + a * (Math.cos(theta) + triangularity * Math.cos(2 * theta))
        const Z_plasma = a * elongation * Math.sin(theta)
        // Convert to canvas coordinates (R is horizontal/x, Z is vertical/y)
        const x = worldToCanvasX(R_plasma)
        const y = worldToCanvasY(Z_plasma)
        if (i === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      }
      ctx.closePath()
      ctx.stroke()
      ctx.globalAlpha = 1.0

      // Violation severity indicator ring (from history)
      if (violationHistoryRef.current.length > 5) {
        const nPoints = violationHistoryRef.current.length
        const angles = Array.from({ length: nPoints }, (_, i) => (i / nPoints) * 2 * Math.PI)
        
        // Python: base_radius = plasma_boundary_radius * 0.6
        const baseRadiusRing = plasmaBoundaryRadius * 0.6
        const radii = violationHistoryRef.current.map(v => {
          // Python: radii = base_radius + np.array(self.violation_history) * 1.2
          const r = baseRadiusRing + v * 1.2
          // Python: radii = np.clip(radii, base_radius * 0.4, plasma_boundary_radius * 0.9)
          return Math.max(baseRadiusRing * 0.4, Math.min(r, plasmaBoundaryRadius * 0.9))
        })

        // Convert to x, y coordinates (Python: trend_x = radii * np.cos(angles))
        const trend_x = radii.map((r, i) => r * Math.cos(angles[i]))
        const trend_y = radii.map((r, i) => r * Math.sin(angles[i]))
        const base_x = Array.from({ length: nPoints }, (_, i) => baseRadiusRing * Math.cos(angles[i]))
        const base_y = Array.from({ length: nPoints }, (_, i) => baseRadiusRing * Math.sin(angles[i]))

        // Draw ring
        ctx.strokeStyle = statusColor
        ctx.lineWidth = 2 * scale
        ctx.globalAlpha = 0.6
        ctx.beginPath()
        for (let i = 0; i < angles.length; i++) {
          const x = worldToCanvasX(trend_x[i])
          const y = worldToCanvasY(trend_y[i])
          if (i === 0) {
            ctx.moveTo(x, y)
          } else {
            ctx.lineTo(x, y)
          }
        }
        ctx.closePath()
        ctx.stroke()

        // Fill area
        ctx.fillStyle = statusColor
        ctx.globalAlpha = 0.1
        for (let i = 0; i < angles.length - 1; i++) {
          const baseX1 = worldToCanvasX(base_x[i])
          const baseY1 = worldToCanvasY(base_y[i])
          const baseX2 = worldToCanvasX(base_x[i + 1])
          const baseY2 = worldToCanvasY(base_y[i + 1])
          const trendX1 = worldToCanvasX(trend_x[i])
          const trendY1 = worldToCanvasY(trend_y[i])
          const trendX2 = worldToCanvasX(trend_x[i + 1])
          const trendY2 = worldToCanvasY(trend_y[i + 1])
          
          ctx.beginPath()
          ctx.moveTo(baseX1, baseY1)
          ctx.lineTo(baseX2, baseY2)
          ctx.lineTo(trendX2, trendY2)
          ctx.lineTo(trendX1, trendY1)
          ctx.closePath()
          ctx.fill()
        }
        ctx.globalAlpha = 1.0

        // Trend indicator
        if (violationHistoryRef.current.length > 10) {
          const recent = violationHistoryRef.current.slice(-5)
          const older = violationHistoryRef.current.slice(-15, -5)
          const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length
          const olderAvg = older.reduce((a, b) => a + b, 0) / older.length
          
          if (recentAvg < olderAvg) {
            ctx.fillStyle = "lime"
            ctx.font = `bold ${9 * scale}px monospace`
            ctx.textAlign = "center"
            ctx.fillText("↓ Improving", worldToCanvasX(0), worldToCanvasY(-11))
          } else if (recentAvg > olderAvg) {
            ctx.fillStyle = "red"
            ctx.font = `bold ${9 * scale}px monospace`
            ctx.textAlign = "center"
            ctx.fillText("↑ Worsening", worldToCanvasX(0), worldToCanvasY(-11))
          }
        }
      }

      // Parameter bars on the right (same as Python)
      // Python: bar_x = 10.5, bar_y = -8 + i * bar_spacing
      const barX = worldToCanvasX(10.5)
      const barWidth = 0.8 * scale
      const barSpacing = 1.2 * scale
      const safeBounds = {
        beta_N: [0.5, 3.0],
        q_min: [1.0, Infinity],
        q95: [3.0, 5.0]
      }

      const params = [
        { name: "beta_N", value: data.beta_N, min: safeBounds.beta_N[0], max: safeBounds.beta_N[1], color: "cyan", label: "β_N: Pressure" },
        { name: "q_min", value: data.q_min, min: safeBounds.q_min[0], max: safeBounds.q_min[1], color: "green", label: "q_min: Internal stability" },
        { name: "q95", value: data.q95, min: safeBounds.q95[0], max: safeBounds.q95[1], color: "blue", label: "q95: Edge stability" }
      ]

      params.forEach((param, i) => {
        const barY = worldToCanvasY(-8 + i * 1.2)  // Python: bar_y = -8 + i * bar_spacing
        const inBounds = param.max === Infinity 
          ? param.value >= param.min 
          : param.value >= param.min && param.value <= param.max
        
        const barColor = inBounds ? "#00ff00" : "#ff0000"
        
        // Normalize value for display
        let displayVal: number
        if (param.max === Infinity) {
          displayVal = (param.value - param.min) / 1.0
          displayVal = Math.min(displayVal, 2.0)
        } else {
          displayVal = (param.value - param.min) / (param.max - param.min)
          displayVal = Math.max(0, Math.min(displayVal, 1.5))
        }

        // Draw safe range background
        const safeHeight = 0.3 * scale
        ctx.fillStyle = "#004400"
        ctx.strokeStyle = "#00aa00"
        ctx.lineWidth = 1
        ctx.globalAlpha = 0.5
        ctx.fillRect(barX, barY - safeHeight / 2, barWidth, safeHeight)
        ctx.strokeRect(barX, barY - safeHeight / 2, barWidth, safeHeight)
        ctx.globalAlpha = 1.0

        // Draw current value indicator
        const indicatorY = barY - safeHeight / 2 + (displayVal / 1.5) * safeHeight
        ctx.fillStyle = barColor
        ctx.strokeStyle = "white"
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc(barX + barWidth / 2, indicatorY, 0.1 * scale, 0, 2 * Math.PI)
        ctx.fill()
        ctx.stroke()

        // Arrow showing direction
        const prevKey = `prev_${param.name}` as keyof typeof prevValuesRef.current
        const prevVal = prevValuesRef.current[prevKey]
        if (prevVal !== undefined) {
          const arrowColor = param.value > prevVal 
            ? (inBounds ? "#00ff00" : "#ff0000")
            : (inBounds ? "#ff0000" : "#00ff00")
          
          ctx.strokeStyle = arrowColor
          ctx.fillStyle = arrowColor
          ctx.lineWidth = 2
          const arrowX = barX + barWidth + 0.2 * scale
          const arrowLen = 0.3 * scale
          
          if (param.value > prevVal) {
            // Arrow right
            ctx.beginPath()
            ctx.moveTo(arrowX, indicatorY)
            ctx.lineTo(arrowX + arrowLen, indicatorY)
            ctx.lineTo(arrowX + arrowLen - 0.1 * scale, indicatorY - 0.05 * scale)
            ctx.moveTo(arrowX + arrowLen, indicatorY)
            ctx.lineTo(arrowX + arrowLen - 0.1 * scale, indicatorY + 0.05 * scale)
            ctx.stroke()
          } else if (param.value < prevVal) {
            // Arrow left
            ctx.beginPath()
            ctx.moveTo(arrowX, indicatorY)
            ctx.lineTo(arrowX - arrowLen, indicatorY)
            ctx.lineTo(arrowX - arrowLen + 0.1 * scale, indicatorY - 0.05 * scale)
            ctx.moveTo(arrowX - arrowLen, indicatorY)
            ctx.lineTo(arrowX - arrowLen + 0.1 * scale, indicatorY + 0.05 * scale)
            ctx.stroke()
          }
        }
        prevValuesRef.current[prevKey] = param.value

        // Label (Python: bar_y - 0.5)
        ctx.fillStyle = "white"
        ctx.font = `bold ${9 * scale}px monospace`
        ctx.textAlign = "center"
        ctx.fillText(param.label, barX + barWidth / 2, worldToCanvasY(-8 + i * 1.2 - 0.5))
      })

      // Status circle on the left (same as Python)
      // Python: scatter(-10.5, 0, s=status_circle_size)
      // Python uses scatter size which is area-based: s=80 means area=80, so radius ≈ sqrt(80/π) ≈ 5.05
      const statusCircleSize = data.ok ? 80 : (data.is_corrective ? 100 : 60)
      const statusCircleColor = data.ok ? "#00ff00" : (data.is_corrective ? "#ff8800" : "#ff0000")
      const statusX = worldToCanvasX(-10.5)
      const statusY = worldToCanvasY(0)
      // Convert scatter size to radius (area = π * r², so r = sqrt(area/π))
      const statusRadius = Math.sqrt(statusCircleSize / Math.PI) * scale

      ctx.fillStyle = statusCircleColor
      ctx.strokeStyle = "white"
      ctx.lineWidth = 3 * scale
      ctx.globalAlpha = 0.8
      ctx.beginPath()
      ctx.arc(statusX, statusY, statusRadius, 0, 2 * Math.PI)
      ctx.fill()
      ctx.stroke()
      ctx.globalAlpha = 1.0

      // Status symbol
      const symbol = data.ok ? "✓" : (data.is_corrective ? "↻" : "✗")
      ctx.fillStyle = "white"
      ctx.font = `bold ${30 * scale}px monospace`
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText(symbol, statusX, statusY)

      // Main status text at bottom with background box (like Python)
      // Python: text(0, -9, status_text)
      const statusText = `${status} | β_N=${data.beta_N.toFixed(2)} | q_min=${data.q_min.toFixed(2)} | q95=${data.q95.toFixed(2)}`
      const textX = worldToCanvasX(0)
      const textY = worldToCanvasY(-9)
      
      // Draw background box (rounded rectangle)
      ctx.font = `bold ${12 * scale}px monospace`
      ctx.textAlign = "center"
      const metrics = ctx.measureText(statusText)
      const textWidth = metrics.width
      const textHeight = 12 * scale * 1.2
      const padding = 8 * scale
      const cornerRadius = 5 * scale
      const boxX = textX - textWidth / 2 - padding
      const boxY = textY - textHeight / 2
      const boxWidth = textWidth + padding * 2
      const boxHeight = textHeight + padding
      
      ctx.fillStyle = "rgba(0, 0, 0, 0.8)"
      ctx.strokeStyle = statusColor
      ctx.lineWidth = 2 * scale
      ctx.beginPath()
      // Draw rounded rectangle manually
      ctx.moveTo(boxX + cornerRadius, boxY)
      ctx.lineTo(boxX + boxWidth - cornerRadius, boxY)
      ctx.quadraticCurveTo(boxX + boxWidth, boxY, boxX + boxWidth, boxY + cornerRadius)
      ctx.lineTo(boxX + boxWidth, boxY + boxHeight - cornerRadius)
      ctx.quadraticCurveTo(boxX + boxWidth, boxY + boxHeight, boxX + boxWidth - cornerRadius, boxY + boxHeight)
      ctx.lineTo(boxX + cornerRadius, boxY + boxHeight)
      ctx.quadraticCurveTo(boxX, boxY + boxHeight, boxX, boxY + boxHeight - cornerRadius)
      ctx.lineTo(boxX, boxY + cornerRadius)
      ctx.quadraticCurveTo(boxX, boxY, boxX + cornerRadius, boxY)
      ctx.closePath()
      ctx.fill()
      ctx.stroke()
      
      // Draw text
      ctx.fillStyle = statusColor
      ctx.textBaseline = "middle"
      ctx.fillText(statusText, textX, textY)

      timeRef.current += 0.1
      animationFrameRef.current = requestAnimationFrame(draw)
    }

    // Handle resize
    const resizeCanvas = () => {
      const container = canvas.parentElement
      if (container) {
        const rect = container.getBoundingClientRect()
        canvas.width = rect.width
        canvas.height = rect.height
      }
    }

    resizeCanvas()
    const resizeObserver = new ResizeObserver(() => {
      resizeCanvas()
    })
    
    if (canvas.parentElement) {
      resizeObserver.observe(canvas.parentElement)
    }
    
    window.addEventListener("resize", resizeCanvas)
    draw()

    return () => {
      window.removeEventListener("resize", resizeCanvas)
      resizeObserver.disconnect()
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [data])

  return (
    <div className="relative w-full">
      <div className="relative aspect-square w-full overflow-hidden rounded-lg border border-foreground/20 bg-foreground/5 backdrop-blur-md">
        <canvas
          ref={canvasRef}
          className="h-full w-full"
          style={{ imageRendering: "crisp-edges", display: "block" }}
          width={800}
          height={800}
        />
        
        {!connected && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="text-center">
              <div className="mb-4 text-foreground/50">
                <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-foreground/20 border-t-foreground"></div>
              </div>
              <p className="font-mono text-sm text-foreground/70">
                {error || "Connecting to visualization server..."}
              </p>
              {error && (
                <p className="mt-2 font-mono text-xs text-foreground/50">
                  Run: python chamber_websocket_server.py
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
