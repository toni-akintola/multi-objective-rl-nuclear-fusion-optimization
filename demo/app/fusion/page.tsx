"use client"

import { Shader, ChromaFlow, Swirl } from "shaders/react"
import { CustomCursor } from "@/components/custom-cursor"
import { GrainOverlay } from "@/components/grain-overlay"
import { FusionHeroSection } from "@/components/sections/fusion-hero"
import { ProblemSection } from "@/components/sections/problem-section"
import { SolutionSection } from "@/components/sections/solution-section"
import { ApproachSection } from "@/components/sections/approach-section"
import { PlasmaSection } from "@/components/sections/plasma-section"
import { Navigation } from "@/components/navigation"
import { useRef, useEffect, useState, useCallback } from "react"
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
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const [currentSection, setCurrentSection] = useState(0)
  const [isLoaded, setIsLoaded] = useState(false)
  const touchStartY = useRef(0)
  const touchStartX = useRef(0)
  const shaderContainerRef = useRef<HTMLDivElement>(null)
  const scrollThrottleRef = useRef<number | undefined>(undefined)
  const sections = [FusionHeroSection, ProblemSection, SolutionSection, ApproachSection, PlasmaSection]
  const searchParams = useSearchParams()

  // Read section from URL on mount
  useEffect(() => {
    const sectionParam = searchParams.get('section')
    if (sectionParam) {
      const sectionIndex = parseInt(sectionParam, 10)
      if (sectionIndex >= 0 && sectionIndex < sections.length) {
        setCurrentSection(sectionIndex)
        // Scroll to section
        if (scrollContainerRef.current) {
          const sectionHeight = scrollContainerRef.current.offsetHeight
          scrollContainerRef.current.scrollTo({
            top: sectionHeight * sectionIndex,
            behavior: "instant",
          })
        }
      }
    }
  }, [searchParams, sections.length])

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

  const scrollToSection = useCallback((index: number) => {
    if (scrollContainerRef.current && index !== currentSection) {
      const container = scrollContainerRef.current
      const sectionHeight = container.offsetHeight
      const targetTop = sectionHeight * index
      
      // Add fade-out effect to current section
      const currentSectionElement = container.children[currentSection] as HTMLElement
      if (currentSectionElement) {
        currentSectionElement.style.transition = "opacity 0.3s ease-out, transform 0.5s cubic-bezier(0.4, 0, 0.2, 1)"
        currentSectionElement.style.opacity = "0.3"
        currentSectionElement.style.transform = `scale(0.95) translateY(${index > currentSection ? '-20px' : '20px'})`
      }
      
      // Smooth scroll with custom easing
      const startTop = container.scrollTop
      const distance = targetTop - startTop
      const duration = 800 // milliseconds
      const startTime = performance.now()
      
      const easeInOutCubic = (t: number): number => {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
      }
      
      const animateScroll = (currentTime: number) => {
        const elapsed = currentTime - startTime
        const progress = Math.min(elapsed / duration, 1)
        const eased = easeInOutCubic(progress)
        
        container.scrollTop = startTop + distance * eased
        
        if (progress < 1) {
          requestAnimationFrame(animateScroll)
        } else {
          // Fade in new section
          const newSectionElement = container.children[index] as HTMLElement
          if (newSectionElement) {
            newSectionElement.style.transition = "opacity 0.5s ease-in, transform 0.6s cubic-bezier(0.4, 0, 0.2, 1)"
            newSectionElement.style.opacity = "0"
            newSectionElement.style.transform = `scale(1.05) translateY(${index > currentSection ? '30px' : '-30px'})`
            
            // Trigger reflow
            newSectionElement.offsetHeight
            
            requestAnimationFrame(() => {
              newSectionElement.style.opacity = "1"
              newSectionElement.style.transform = "scale(1) translateY(0)"
              
              // Reset after animation
              setTimeout(() => {
                newSectionElement.style.transition = ""
                newSectionElement.style.transform = ""
                newSectionElement.style.opacity = ""
              }, 600)
            })
          }
          
          // Reset current section after animation
          if (currentSectionElement) {
            setTimeout(() => {
              currentSectionElement.style.transition = ""
              currentSectionElement.style.transform = ""
              currentSectionElement.style.opacity = ""
            }, 500)
          }
        }
      }
      
      requestAnimationFrame(animateScroll)
      setCurrentSection(index)
      
      // Update URL with section parameter
      const url = new URL(window.location.href)
      url.searchParams.set('section', index.toString())
      window.history.pushState({}, '', url)
    }
  }, [currentSection])

  // Disable wheel scrolling
  useEffect(() => {
    const handleWheel = (e: WheelEvent) => {
      e.preventDefault()
    }

    const container = scrollContainerRef.current
    if (container) {
      container.addEventListener("wheel", handleWheel, { passive: false })
    }

    return () => {
      if (container) {
        container.removeEventListener("wheel", handleWheel)
      }
    }
  }, [])

  // Disable touch scrolling
  useEffect(() => {
    const handleTouchMove = (e: TouchEvent) => {
      e.preventDefault()
    }

    const container = scrollContainerRef.current
    if (container) {
      container.addEventListener("touchmove", handleTouchMove, { passive: false })
    }

    return () => {
      if (container) {
        container.removeEventListener("touchmove", handleTouchMove)
      }
    }
  }, [])

  // Arrow key navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault()
        
        if (e.key === "ArrowDown" && currentSection < sections.length - 1) {
          scrollToSection(currentSection + 1)
        } else if (e.key === "ArrowUp" && currentSection > 0) {
          scrollToSection(currentSection - 1)
        }
      }
    }

    window.addEventListener("keydown", handleKeyDown)

    return () => {
      window.removeEventListener("keydown", handleKeyDown)
    }
  }, [currentSection, sections.length, scrollToSection])


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

      <Navigation
        isLoaded={isLoaded}
        currentSection={currentSection}
        onSectionClick={scrollToSection}
        variant="fusion"
      />

      <div
        ref={scrollContainerRef}
        data-scroll-container
        className={`relative z-10 flex flex-col h-screen overflow-y-auto overflow-x-hidden transition-opacity duration-700 ${
          isLoaded ? "opacity-100" : "opacity-0"
        }`}
        style={{ 
          scrollbarWidth: "none", 
          msOverflowStyle: "none",
          scrollBehavior: "smooth"
        }}
      >
        {sections.map((Section, index) => (
          <div 
            key={index} 
            className="h-screen w-screen shrink-0"
            style={{ scrollSnapAlign: "start", scrollSnapStop: "always" }}
          >
            <Section />
          </div>
        ))}
      </div>

      <style jsx global>{`
        div::-webkit-scrollbar {
          display: none;
        }
      `}</style>
    </main>
  )
}