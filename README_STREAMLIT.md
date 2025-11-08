# Interactive Web Visualization

## No TypeScript Needed! 🎉

This project includes a **Streamlit web app** that provides an interactive frontend - all in Python!

## How to Run

1. **Install dependencies:**
   ```bash
   pip install streamlit pandas
   # or if using uv:
   uv sync
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - Or manually navigate to the URL shown in the terminal

## Features

- **Interactive Controls**: Adjust shape penalty, damping, episodes, and steps
- **Real-time Visualization**: See shape parameters, severity, and trajectory
- **Step-by-Step Table**: View detailed data for each step
- **2D Trajectory Plot**: Watch the plasma move toward safety
- **Statistics Dashboard**: Key metrics at a glance

## Alternative: TypeScript Frontend

If you prefer a TypeScript/React frontend:

1. **Backend**: Create a FastAPI/Flask API that runs the simulation
2. **Frontend**: React + TypeScript + D3.js/Plotly.js for visualizations
3. **Communication**: WebSockets for real-time updates

But Streamlit is much faster to set up! 🚀

