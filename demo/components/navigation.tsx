"use client"

import Link from "next/link"

interface NavigationProps {
  isLoaded: boolean
  currentSection?: number
  onSectionClick?: (index: number) => void
  variant?: "home" | "fusion"
}

export function Navigation({ isLoaded, currentSection = 0, onSectionClick, variant = "home" }: NavigationProps) {
  const logoConfig = variant === "fusion" 
    ? { symbol: "ϕ", title: "Fusion Lab" }
    : { symbol: "ϕ", title: "Fusion Lab" }

  const navItems = [
    { label: "Problem", type: "section", index: 1 },
    { label: "Solution", type: "section", index: 2 },
    { label: "Approach", type: "section", index: 3 },
    { label: "Plasma", type: "section", index: 4 },
  ]

  return (
    <nav
      className={`fixed left-0 right-0 top-0 z-50 flex items-center justify-between px-6 py-6 transition-opacity duration-700 md:px-12 ${
        isLoaded ? "opacity-100" : "opacity-0"
      }`}
    >
      <button
        onClick={() => onSectionClick?.(0)}
        className="flex items-center gap-2 transition-transform hover:scale-105"
      >
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/15 backdrop-blur-md transition-all duration-300 hover:scale-110 hover:bg-foreground/25">
          <span className="font-sans text-xl font-bold text-foreground">{logoConfig.symbol}</span>
        </div>
        <span className="font-sans text-xl font-semibold tracking-tight text-foreground">{logoConfig.title}</span>
      </button>

      <div className="hidden items-center gap-8 md:flex">
        {navItems.map((item) => {
          if (item.type === "link" && "href" in item && item.href) {
            return (
              <Link
                key={item.label}
                href={item.href}
                className="group relative font-sans text-sm font-medium transition-colors text-foreground/80 hover:text-foreground"
              >
                {item.label}
                <span className="absolute -bottom-1 left-0 h-px bg-foreground transition-all duration-300 w-0 group-hover:w-full" />
              </Link>
            )
          }

          return (
            <button
              key={item.label}
              onClick={() => onSectionClick?.(item.index ?? 0)}
              className={`group relative font-sans text-sm font-medium transition-colors ${
                currentSection === item.index ? "text-foreground" : "text-foreground/80 hover:text-foreground"
              }`}
            >
              {item.label}
              <span
                className={`absolute -bottom-1 left-0 h-px bg-foreground transition-all duration-300 ${
                  currentSection === item.index ? "w-full" : "w-0 group-hover:w-full"
                }`}
              />
            </button>
          )
        })}
      </div>

    </nav>
  )
}
