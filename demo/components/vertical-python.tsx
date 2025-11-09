"use client"

import { MagneticButton } from "@/components/magnetic-button"

export function VerticalPython() {
  const handleLaunchVisualization = async () => {
    try {
      const response = await fetch("/api/launch-vertical")
      const data = await response.json()
      if (data.success) {
        alert("✅ Rotating 3D vertical guard visualization launched! A matplotlib window should open showing plasma vertical position with interactive coil disable/enable controls.")
      } else {
        alert(`Error: ${data.error}`)
      }
    } catch (error) {
      alert("Failed to launch visualization. Make sure Python and dependencies are installed.")
    }
  }

  return (
    <div>
      <MagneticButton
        size="lg"
        variant="primary"
        onClick={handleLaunchVisualization}
      >
        🐍 Launch Python Visualization
      </MagneticButton>
      <p className="mt-4 font-mono text-xs text-foreground/50">
        Opens a real-time matplotlib window showing vertical position monitoring
      </p>
    </div>
  )
}

