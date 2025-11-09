"use client"

import { CustomCursor } from "@/components/custom-cursor"
import { GrainOverlay } from "@/components/grain-overlay"
import { FusionHeroSection } from "@/components/sections/fusion-hero"

export default function Home() {
  return (
    <main className="relative min-h-screen w-full bg-gradient-to-br from-slate-950 via-blue-950 to-slate-900">
      <CustomCursor />
      <GrainOverlay />
      
      <FusionHeroSection />
    </main>
  )
}