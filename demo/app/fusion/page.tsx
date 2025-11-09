"use client"

import { Shader, ChromaFlow, Swirl } from "shaders/react"
import { CustomCursor } from "@/components/custom-cursor"
import { GrainOverlay } from "@/components/grain-overlay"
import { FusionHeroSection } from "@/components/sections/fusion-hero"
import { ProblemSection } from "@/components/sections/problem-section"
import { SolutionSection } from "@/components/sections/solution-section"
import { ApproachSection } from "@/components/sections/approach-section"
import { MagneticButton } from "@/components/magnetic-button"
import { useRef, useEffect, useState } from "react"
import { ChevronDown, ChevronUp } from "lucide-react"
import Link from "next/link"
import { useSearchParams } from "next/navigation"
import { Suspense } from "react"

export default function FusionPage() {
  return (
    <Suspense fallback={<div className="h-screen w-full bg-background" />}>
      <FusionPageContent />
    </Suspense>
  )
}

function FusionPageContent() {
  const [currentSection, setCurrentSection] = useState(0)
  const [isLoaded, setIsLoaded] = useState(false)
  const shaderContainerRef = useRef<HTMLDivElement>(null)
  const sections = [FusionHeroSection, ProblemSection, SolutionSection, ApproachSection]
  const searchParams = useSearchParams()

  // Read section from URL on mount
  useEffect(() => {
    const sectionParam = searchParams.get('section')
    if (sectionParam) {
      const sectionIndex = parseInt(sectionParam, 10)
      if (sectionIndex >= 0 && sectionIndex < sections.length) {
        setCurrentSection(sectionIndex)
      }
    }
  }, [searchParams])

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

  const scrollToSection = (index: number) => {
    if (index >= 0 && index < sections.length) {
      setCurrentSection(index)
    }
  }

  const goToNextSection = () => {
    if (currentSection < sections.length - 1) {
      setCurrentSection(currentSection + 1)
    }
  }

  const goToPrevSection = () => {
    if (currentSection > 0) {
      setCurrentSection(currentSection - 1)
    }
  }

  return (
    <main className="relative h-screen w-full overflow-hidden bg-background">
      <CustomCursor />
      <GrainOverlay />

      <div
        ref={shaderContainerRef}
        className={`fixed inset-0 z-0 transition-opacity duration-700 ${isLoaded ? "opacity-100" : "opacity-0"}`}
        style={{ contain: "strict" }}
      >
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
        <div className="absolute inset-0 bg-black/20" />
      </div>

      <nav
        className={`fixed left-0 right-0 top-0 z-50 flex items-center justify-between px-6 py-6 transition-opacity duration-700 md:px-12 ${
          isLoaded ? "opacity-100" : "opacity-0"
        }`}
      >
        <button
          onClick={() => scrollToSection(0)}
          className="flex items-center gap-2 transition-transform hover:scale-105"
        >
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/15 backdrop-blur-md transition-all duration-300 hover:scale-110 hover:bg-foreground/25">
            <span className="font-sans text-xl font-bold text-foreground">ϕ</span>
          </div>
          <span className="font-sans text-xl font-semibold tracking-tight text-foreground">Fusion Lab</span>
        </button>

        <div className="hidden items-center gap-8 md:flex">
          {["Problem", "Solution", "Approach", "Plasma", "Vertical", "Insights"].map((item, index) => {
            // Plasma links to chamber page, Vertical links to vertical-vis page
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
            if (item === "Vertical") {
              return (
                <Link
                  key={item}
                  href="/vertical-vis"
                  className="group relative font-sans text-sm font-medium transition-colors text-foreground/80 hover:text-foreground"
                >
                  {item}
                  <span className="absolute -bottom-1 left-0 h-px bg-foreground transition-all duration-300 w-0 group-hover:w-full" />
                </Link>
              )
            }
            // Map to correct sections: Problem=1 (ProblemSection), Solution=2 (SolutionSection), Approach=3 (ApproachSection), Insights=3 (ApproachSection)
            const sectionMap: Record<string, number> = {
              "Problem": 1,
              "Solution": 2,
              "Approach": 3,
              "Insights": 3
            }
            const sectionIndex = sectionMap[item] ?? 0
            return (
              <button
                key={item}
                onClick={() => scrollToSection(sectionIndex)}
                className={`group relative font-sans text-sm font-medium transition-colors ${
                  currentSection === sectionIndex ? "text-foreground" : "text-foreground/80 hover:text-foreground"
                }`}
              >
                {item}
                <span
                  className={`absolute -bottom-1 left-0 h-px bg-foreground transition-all duration-300 ${
                    currentSection === sectionIndex ? "w-full" : "w-0 group-hover:w-full"
                  }`}
                />
              </button>
            )
          })}
        </div>
      </nav>

      <div
        className={`relative z-10 h-screen overflow-y-auto pt-20 transition-opacity duration-700 ${
          isLoaded ? "opacity-100" : "opacity-0"
        }`}
      >
        {sections.map((Section, index) => (
          <div
            key={index}
            className={`${currentSection === index ? "block" : "hidden"}`}
          >
            <Section />
          </div>
        ))}
      </div>

      {/* Navigation Arrows */}
      {currentSection > 0 && (
        <button
          onClick={goToPrevSection}
          className={`fixed left-1/2 top-8 z-50 flex h-10 w-10 -translate-x-1/2 items-center justify-center rounded-full bg-foreground/10 backdrop-blur-md transition-all duration-300 hover:bg-foreground/20 ${
            isLoaded ? "opacity-100" : "opacity-0"
          }`}
          aria-label="Previous section"
        >
          <ChevronUp className="h-5 w-5 text-foreground" />
        </button>
      )}

      {currentSection < sections.length - 1 && (
        <button
          onClick={goToNextSection}
          className={`fixed bottom-8 left-1/2 z-50 flex h-10 w-10 -translate-x-1/2 items-center justify-center rounded-full bg-foreground/10 backdrop-blur-md transition-all duration-300 hover:bg-foreground/20 ${
            isLoaded ? "opacity-100" : "opacity-0"
          }`}
          aria-label="Next section"
        >
          <ChevronDown className="h-5 w-5 text-foreground" />
        </button>
      )}
    </main>
  )
}