"""
run.py — Single-command launcher for the Smart Traffic Light RL Project.
Starts both:
  1. Streamlit analytics dashboard (browser tab)
  2. Pygame live simulation (animated intro → game window)
Just run:  python run.py
"""
import sys
import os
import subprocess
import time
import webbrowser
import signal

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON       = sys.executable
STREAMLIT    = os.path.join(os.path.dirname(PYTHON), "Scripts", "streamlit.exe")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "trained_q_table.pkl")
STREAMLIT_PORT = 8501
STREAMLIT_URL  = f"http://localhost:{STREAMLIT_PORT}"

# ANSI colours (Windows 10+ supports these)
GREEN  = "\033[92m"
BLUE   = "\033[94m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def banner():
    print(f"""
{BOLD}{GREEN}╔══════════════════════════════════════════════════════════╗
║       Smart Traffic Light Controller — RL Project        ║
║    Tabular Q-Learning | Multi-Tier Emergency Priority    ║
╚══════════════════════════════════════════════════════════╝{RESET}
""")

def check_model():
    if not os.path.exists(MODEL_PATH):
        print(f"{YELLOW}⚠  No trained model found at {MODEL_PATH}{RESET}")
        print(f"{YELLOW}   Training the RL agent first (~2.5 minutes)...{RESET}\n")
        result = subprocess.run(
            [PYTHON, "-u", "main.py", "--train"],
            cwd=PROJECT_ROOT
        )
        if result.returncode != 0:
            print(f"{RED}✗  Training failed. Check errors above.{RESET}")
            sys.exit(1)
        print(f"\n{GREEN}✓  Model trained and saved.{RESET}\n")

def start_streamlit():
    print(f"{BLUE}▶  Starting Streamlit dashboard on {STREAMLIT_URL} ...{RESET}")
    # Use streamlit.exe if available, otherwise python -m streamlit
    if os.path.exists(STREAMLIT):
        cmd = [STREAMLIT, "run",
               os.path.join(PROJECT_ROOT, "streamlit_app", "app.py"),
               "--server.port", str(STREAMLIT_PORT),
               "--server.headless", "true",
               "--browser.gatherUsageStats", "false"]
    else:
        cmd = [PYTHON, "-m", "streamlit", "run",
               os.path.join(PROJECT_ROOT, "streamlit_app", "app.py"),
               "--server.port", str(STREAMLIT_PORT),
               "--server.headless", "true",
               "--browser.gatherUsageStats", "false"]

    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    )
    return proc

def wait_for_streamlit(timeout=12):
    """Poll until Streamlit is accepting connections."""
    import urllib.request, urllib.error
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(STREAMLIT_URL, timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False

def open_browser():
    print(f"{GREEN}✓  Streamlit ready — opening browser tab...{RESET}")
    webbrowser.open(STREAMLIT_URL)

def start_pygame():
    print(f"{BLUE}▶  Launching Pygame simulation (animated intro screen)...{RESET}\n")
    result = subprocess.run(
        [PYTHON, "-u", "main.py", "--simulate"],
        cwd=PROJECT_ROOT
    )
    return result.returncode

def main():
    # Enable ANSI on Windows
    if sys.platform == "win32":
        os.system("color")

    banner()

    # Step 1: Ensure model exists
    check_model()

    # Step 2: Start Streamlit in background
    st_proc = start_streamlit()
    print(f"   Waiting for Streamlit to start", end="", flush=True)
    for _ in range(15):
        time.sleep(0.8)
        print(".", end="", flush=True)
        try:
            import urllib.request
            urllib.request.urlopen(STREAMLIT_URL, timeout=1)
            break
        except Exception:
            pass
    print()
    open_browser()
    print()

    print(f"{YELLOW}ℹ  Streamlit dashboard is live at: {STREAMLIT_URL}{RESET}")
    print(f"{YELLOW}ℹ  Pygame window will open next — use it for the live simulation.{RESET}")
    print(f"{YELLOW}ℹ  Close the Pygame window to stop everything.\n{RESET}")

    # Step 3: Run Pygame (blocks until user closes the window)
    exit_code = start_pygame()

    # Step 4: Clean up Streamlit
    print(f"\n{YELLOW}⏹  Pygame closed — shutting down Streamlit server...{RESET}")
    if sys.platform == "win32":
        st_proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        st_proc.terminate()
    st_proc.wait(timeout=5)
    print(f"{GREEN}✓  All services stopped. Goodbye!{RESET}\n")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
