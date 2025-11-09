"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"

interface NavigationProps {
  isLoaded: boolean
  variant?: "home" | "fusion"
}

export function Navigation({ isLoaded, variant = "home" }: NavigationProps) {
  const pathname = usePathname()
  const logoConfig = variant === "fusion" 
    ? { symbol: "Q", title: "QStable" }
    : { symbol: "Q", title: "QStable" }

  const navItems = [
    { label: "Problem", type: "link", href: "/problem" },
    { label: "Solution", type: "link", href: "/solution" },
    { label: "Approach", type: "link", href: "/approach" },
    { label: "Plasma", type: "link", href: "/plasma" },
    { label: "References", type: "link", href: "/references" },
  ]

  return (
    <nav
      className={`absolute left-0 right-0 top-0 z-50 flex items-center justify-between px-6 py-6 transition-opacity duration-700 md:px-12 ${
        isLoaded ? "opacity-100" : "opacity-0"
      }`}
    >
      <Link
        href="/"
        className="flex items-center gap-2 transition-transform hover:scale-105"
      >
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-foreground/15 backdrop-blur-md transition-all duration-300 hover:scale-110 hover:bg-foreground/25">
          <span className="font-sans text-xl font-bold text-foreground">{logoConfig.symbol}</span>
        </div>
        <span className="font-sans text-xl font-semibold tracking-tight text-foreground">{logoConfig.title}</span>
      </Link>

      <div className="hidden items-center gap-8 md:flex">
        {navItems.map((item) => {
          if (item.type === "link" && "href" in item && item.href) {
            const isActive = pathname === item.href
            return (
              <Link
                key={item.label}
                href={item.href}
                className={`group relative font-sans text-sm font-medium transition-colors ${
                  isActive ? "text-foreground" : "text-foreground/80 hover:text-foreground"
                }`}
              >
                {item.label}
                <span className={`absolute -bottom-1 left-0 h-px bg-foreground transition-all duration-300 ${
                  isActive ? "w-full" : "w-0 group-hover:w-full"
                }`} />
              </Link>
            )
          }

          return null
        })}
      </div>

    </nav>
  )
}
