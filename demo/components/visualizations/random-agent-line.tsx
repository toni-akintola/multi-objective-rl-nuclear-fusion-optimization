"use client"

import { useEffect, useMemo, useState } from "react"
import { ResponsiveLine } from "@nivo/line"

type EpisodeMetrics = {
  index: number
  reward: number
  length?: number
}

type RunStatistics = {
  reward_mean?: number
  reward_std?: number
  reward_min?: number
  reward_max?: number
  length_mean?: number
  length_std?: number
}

type RunPayload = {
  agent?: string
  episodes?: EpisodeMetrics[]
  statistics?: RunStatistics
}

interface AgentLineChartProps {
  dataPath: string
  fallbackCommand: string
  emptyHint?: string
  seriesName?: string
}

export function AgentLineChart({
  dataPath,
  fallbackCommand,
  emptyHint = "Export a run to populate this chart.",
  seriesName,
}: AgentLineChartProps) {
  const [payload, setPayload] = useState<RunPayload | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    const controller = new AbortController()

    async function fetchData() {
      setIsLoading(true)
      setError(null)

      try {
        const response = await fetch(dataPath, { signal: controller.signal })
        if (!response.ok) {
          throw new Error(response.status === 404 ? "No exported data found yet." : `Request failed (${response.status})`)
        }
        const json = (await response.json()) as RunPayload
        if (active) {
          setPayload(json)
        }
      } catch (err) {
        if (active) {
          if ((err as Error).name !== "AbortError") {
            setError((err as Error).message)
            setPayload(null)
          }
        }
      } finally {
        if (active) {
          setIsLoading(false)
        }
      }
    }

    fetchData()

    return () => {
      active = false
      controller.abort()
    }
  }, [dataPath])

  const chartData = useMemo(() => {
    if (!payload?.episodes || payload.episodes.length === 0) return []

    return [
      {
        id: seriesName ?? payload.agent ?? "Agent",
        data: payload.episodes.map((episode) => ({
          x: episode.index,
          y: episode.reward,
        })),
      },
    ]
  }, [payload, seriesName])

  const theme = useMemo(
    () => ({
      textColor: "#e5e7eb",
      fontSize: 12,
      axis: {
        ticks: {
          text: {
            fill: "#e5e7eb",
          },
        },
        legend: {
          text: {
            fill: "#f3f4f6",
          },
        },
      },
      grid: {
        line: {
          stroke: "rgba(148, 163, 184, 0.15)",
          strokeWidth: 1,
        },
      },
      tooltip: {
        container: {
          background: "rgba(15, 23, 42, 0.92)",
          color: "#f8fafc",
          borderRadius: 10,
          boxShadow: "0 8px 30px rgba(15, 23, 42, 0.18)",
          padding: "8px 12px",
        },
      },
      crosshair: {
        line: {
          stroke: "rgba(248, 250, 252, 0.35)",
          strokeWidth: 1,
          strokeDasharray: "6 6",
        },
      },
    }),
    []
  )

  return (
    <div className="flex h-80 w-full flex-col gap-4">
      <div className="flex-1 overflow-hidden rounded-2xl border border-foreground/15 bg-background/40">
        {isLoading ? (
          <div className="flex h-full items-center justify-center text-sm text-foreground/70">Loading Nivo chart…</div>
        ) : error ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 px-4 text-center text-sm text-foreground/70">
            <p>{error}</p>
            <p className="text-foreground/50">
              Run{" "}
              <code className="rounded bg-foreground/10 px-1">
                {fallbackCommand}
              </code>{" "}
              to generate data.
            </p>
          </div>
        ) : chartData.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 px-4 text-center text-sm text-foreground/70">
            <p>No episode data available.</p>
            <p className="text-foreground/50">{emptyHint}</p>
          </div>
        ) : (
          <ResponsiveLine
            data={chartData}
            theme={theme}
            margin={{ top: 32, right: 40, bottom: 48, left: 64 }}
            xScale={{ type: "linear", min: "auto", max: "auto" }}
            xFormat={(value) => `Episode ${value}`}
            yScale={{ type: "linear", min: "auto", max: "auto", stacked: false, nice: true }}
            yFormat={(value) => `${Number(value).toFixed(2)}`}
            axisBottom={{
              tickSize: 5,
              tickPadding: 8,
              legend: "Episode",
              legendOffset: 36,
              legendPosition: "middle",
              format: (value) => `${value}`,
            }}
            axisLeft={{
              tickSize: 5,
              tickPadding: 8,
              legend: "Episode Reward",
              legendOffset: -50,
              legendPosition: "middle",
              format: (value) => `${Number(value).toFixed(2)}`,
            }}
            enablePoints
            pointSize={9}
            pointColor={{ theme: "background" }}
            pointBorderWidth={2}
            pointBorderColor={{ from: "serieColor" }}
            pointLabel="yFormatted"
            pointLabelYOffset={-12}
            useMesh
            enableArea
            areaOpacity={0.18}
            colors={["#60a5fa"]}
            lineWidth={3}
            curve="monotoneX"
            motionConfig="gentle"
          />
        )}
      </div>

      {payload?.statistics && (
        <div className="flex flex-wrap gap-4 text-xs text-foreground/70">
          <Stat label="Mean Reward" value={payload.statistics.reward_mean} />
          <Stat label="Reward Std" value={payload.statistics.reward_std} />
          <Stat label="Best Reward" value={payload.statistics.reward_max} />
          <Stat label="Worst Reward" value={payload.statistics.reward_min} />
          <Stat label="Mean Length" value={payload.statistics.length_mean} suffix=" steps" />
        </div>
      )}
    </div>
  )
}

export default function RandomAgentLineChart() {
  return (
    <AgentLineChart
      dataPath="/data/random-agent.json"
      fallbackCommand="python main.py --agent random --episodes 10 --export-json demo/public/data/random-agent.json"
      emptyHint="Export a Random agent run to populate this chart."
      seriesName="Random Agent"
    />
  )
}

export function PIDAgentLineChart() {
  return (
    <AgentLineChart
      dataPath="/data/pid-agent.json"
      fallbackCommand="python main.py --agent pid --episodes 10 --export-json demo/public/data/pid-agent.json"
      emptyHint="Export a PID controller run to populate this chart."
      seriesName="PID Controller"
    />
  )
}

interface StatProps {
  label: string
  value?: number
  suffix?: string
}

function Stat({ label, value, suffix = "" }: StatProps) {
  if (value === undefined || Number.isNaN(value)) {
    return null
  }

  const formatted = Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(2)

  return (
    <div className="rounded-full border border-foreground/10 bg-foreground/5 px-3 py-1.5 backdrop-blur-sm">
      <span className="font-medium text-foreground/60">{label}: </span>
      <span className="font-semibold text-foreground/90">
        {formatted}
        {suffix}
      </span>
    </div>
  )
}

