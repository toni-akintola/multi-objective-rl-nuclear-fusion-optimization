# How to Run

## Quick Start

1. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Install dependencies (if not already installed):**
   ```bash
   pip install -e .
   ```
   Or if using `uv`:
   ```bash
   uv sync
   ```

3. **Run the main script:**
   ```bash
   python main.py
   ```

## What It Does

The script will:
- Run 10 episodes with a random agent **without** shape guard (baseline)
- Run 10 episodes with a random agent **with** shape guard enabled
- Generate plots showing reward and episode length comparisons
- Save results to `random_baseline.png`

## Expected Output

You'll see:
- Progress bars for each episode
- Episode statistics (reward, steps)
- Two plots comparing performance with/without shape guard
- Summary statistics printed to console

## Troubleshooting

If you get import errors:
- Make sure the virtual environment is activated: `source .venv/bin/activate`
- Install missing dependencies: `pip install gymnasium gymtorax matplotlib numpy tqdm`

