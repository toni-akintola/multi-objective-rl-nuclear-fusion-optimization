import { NextResponse } from "next/server"
import { spawn } from "child_process"
import path from "path"
import { existsSync } from "fs"

export async function GET() {
  try {
    // Root directory is one level up from demo/
    const rootDir = path.join(process.cwd(), "..")
    
    // Use visualize_vertical_3d.py - the rotating 3D vertical guard visualization with coil controls
    const scriptPath = path.join(rootDir, "visualize_vertical_3d.py")
    
    if (!existsSync(scriptPath)) {
      return NextResponse.json(
        { success: false, error: `Script not found at ${scriptPath}` },
        { status: 404 }
      )
    }
    
    // Try to use venv Python if it exists, otherwise use system python3
    const venvPython = path.join(rootDir, ".venv", "bin", "python")
    const pythonExec = existsSync(venvPython) ? venvPython : "python3"
    
    // Launch the Python script in a detached process
    // Set cwd to root so the script can find dependencies
    const pythonProcess = spawn(pythonExec, [scriptPath], {
      cwd: rootDir,
      detached: true,
      stdio: "ignore",
    })
    
    // Unref so the parent process can exit
    pythonProcess.unref()
    
    return NextResponse.json({ 
      success: true, 
      message: "Rotating 3D vertical guard visualization launched! A matplotlib window should open showing plasma vertical position with interactive coil controls." 
    })
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    )
  }
}

