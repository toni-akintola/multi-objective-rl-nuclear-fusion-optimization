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

export default function Home() {
  return (
    <Suspense fallback={<div className="h-screen w-full bg-background" />}>
      <HomeContent />
    </Suspense>
  )
}

function HomeContent() {
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
    if (scrollContainerRef.current) {
      const sectionHeight = scrollContainerRef.current.offsetHeight
      scrollContainerRef.current.scrollTo({
        top: sectionHeight * index,
        behavior: "smooth",
      })
      setCurrentSection(index)
      
      // Update URL with section parameter
      const url = new URL(window.location.href)
      url.searchParams.set('section', index.toString())
      window.history.pushState({}, '', url)
    }
  }, [])

  useEffect(() => {
    const handleTouchStart = (e: TouchEvent) => {
      touchStartY.current = e.touches[0].clientY
      touchStartX.current = e.touches[0].clientX
    }

    const handleTouchMove = (e: TouchEvent) => {
      if (Math.abs(e.touches[0].clientY - touchStartY.current) > 10) {
        e.preventDefault()
      }
    }

    const handleTouchEnd = (e: TouchEvent) => {
      const touchEndY = e.changedTouches[0].clientY
      const touchEndX = e.changedTouches[0].clientX
      const deltaY = touchStartY.current - touchEndY
      const deltaX = touchStartX.current - touchEndX

      if (Math.abs(deltaY) > Math.abs(deltaX) && Math.abs(deltaY) > 50) {
        if (deltaY > 0 && currentSection < sections.length - 1) {
          scrollToSection(currentSection + 1)
        } else if (deltaY < 0 && currentSection > 0) {
          scrollToSection(currentSection - 1)
        }
      }
    }

    const container = scrollContainerRef.current
    if (container) {
      container.addEventListener("touchstart", handleTouchStart, { passive: true })
      container.addEventListener("touchmove", handleTouchMove, { passive: false })
      container.addEventListener("touchend", handleTouchEnd, { passive: true })
    }

    return () => {
      if (container) {
        container.removeEventListener("touchstart", handleTouchStart)
        container.removeEventListener("touchmove", handleTouchMove)
        container.removeEventListener("touchend", handleTouchEnd)
      }
    }
  }, [currentSection, sections.length, scrollToSection])

  useEffect(() => {
    const handleScroll = () => {
      if (scrollThrottleRef.current) return

      scrollThrottleRef.current = requestAnimationFrame(() => {
        if (!scrollContainerRef.current) {
          scrollThrottleRef.current = undefined
          return
        }

        const sectionHeight = scrollContainerRef.current.offsetHeight
        const scrollTop = scrollContainerRef.current.scrollTop
        const newSection = Math.round(scrollTop / sectionHeight)

        if (newSection !== currentSection && newSection >= 0 && newSection < sections.length) {
          setCurrentSection(newSection)
          
          // Update URL with section parameter
          const url = new URL(window.location.href)
          url.searchParams.set('section', newSection.toString())
          window.history.pushState({}, '', url)
        }

        scrollThrottleRef.current = undefined
      })
    }

    const container = scrollContainerRef.current
    if (container) {
      container.addEventListener("scroll", handleScroll, { passive: true })
    }

    return () => {
      if (container) {
        container.removeEventListener("scroll", handleScroll)
      }
      if (scrollThrottleRef.current) {
        cancelAnimationFrame(scrollThrottleRef.current)
      }
    }
  }, [currentSection, sections.length])



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
        variant="home"
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
          scrollSnapType: "y mandatory",
          scrollBehavior: "smooth",
          WebkitOverflowScrolling: "touch"
        }}
      >
        {sections.map((Section, index) => (
          <div 
            key={index} 
            className="h-screen w-screen shrink-0 flex-shrink-0"
            style={{ scrollSnapAlign: "start", scrollSnapStop: "always", minHeight: "100vh" }}
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