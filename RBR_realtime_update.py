import os
import time
import json
import logging
import shutil
import subprocess
import pandas as pd
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ==================== CONFIGURATION ====================
LOCAL_DAT_FILE = r"D:/Data_bk/RBR_data/RBR_CR1000X_profile_Profile_1sec.dat"
LOCAL_DAT_FILE2 = r"D:/Data_bk/RBR_data/RBR_CR5000_Met_1min.dat"

# GitHub repository settings
GITHUB_REPO_PATH = r"D:/Github/Lab423 website/LAB423-website"   # <-- CHANGE THIS
GITHUB_JSON_NAME = "RBR_CR1000X_profile_Profile_1sec.json"   # Name in the repo

# How often to sync (seconds), update every 5 min
DEBOUNCE_SECONDS = 60*5

# Log file (optional)
LOG_FILE = "push_to_github.log"
# ==================== END CONFIGURATION ====================

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def convert_to_json(dat_file_path,dat_file_path2, resample_seconds=60):
    """Convert .dat file to downsampled JSON for web plotting."""
    try:
        logger.info(f"Converting {dat_file_path} to JSON...")
        # Read .dat file (skip header rows)
        df = pd.read_csv(
            dat_file_path,
            skiprows=[0, 2, 3],          # Adjust if your file structure differs
            delimiter=',',
            quotechar='"',
            parse_dates=[0],
            encoding='utf-8'
        )
        # Clean column names
        df.columns = df.columns.str.strip('"')
        # Ensure TIMESTAMP is datetime
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
        df.set_index('TIMESTAMP', inplace=True)

        # Resample (average) every `resample_seconds` seconds
        measurement_cols = ['WaterT', 'SWR_1', 'SWR_2', 'Diff_SWR']
        df_resampled = df[measurement_cols].resample(f'{resample_seconds}s').mean()
        df_resampled.dropna(inplace=True)

        # Reset index and format timestamp
        df_resampled.reset_index(inplace=True)
        df_resampled['TIMESTAMP'] = df_resampled['TIMESTAMP'].dt.strftime('%Y-%m-%d %H:%M:%S')

        logger.info(f"Read {len(df)} rows, downsampled to {len(df_resampled)} rows")
        # read 1min met measurements named dat_file_path2
        logger.info(f"Converting {dat_file_path2} to JSON...")
        # Read .dat file (skip header rows)
        df_1min_met = pd.read_csv(
            dat_file_path2,
            skiprows=[0, 2, 3],  # Adjust if your file structure differs
            delimiter=',',
            quotechar='"',
            parse_dates=[0],
            encoding='utf-8'
        )
        # Clean column names
        df_1min_met.columns = df_1min_met.columns.str.strip('"')
        # Ensure TIMESTAMP is datetime
        df_1min_met['TIMESTAMP'] = pd.to_datetime(df_1min_met['TIMESTAMP'], errors='coerce')
        logger.info(f"Read {len(df_1min_met)} rows 1min met data")
        df_resampled['TIMESTAMP'] = pd.to_datetime(df_resampled['TIMESTAMP'], errors='coerce')
        df_1min_met['TIMESTAMP'] = pd.to_datetime(df_1min_met['TIMESTAMP'], errors='coerce')
        df_combined = pd.merge(df_resampled, df_1min_met, on='TIMESTAMP', how='left')
        df_combined['TIMESTAMP'] = df_combined['TIMESTAMP'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_combined = df_combined.astype(object).where(pd.notna(df_combined), None) # avoid nan value in df
        # Build JSON
        plot_data = {
            'dates': df_combined['TIMESTAMP'].tolist(),
            'water_temperature': df_combined['WaterT'].tolist(),
            'swr_1': df_combined['SWR_1'].tolist(),
            'swr_2': df_combined['SWR_2'].tolist(),
            'diff_swr': df_combined['Diff_SWR'].tolist(),
            'air_temperature_A': df_combined['AirTC_A_Avg'].tolist(),
            'air_temperature_B': df_combined['AirTC_B_Avg'].tolist(),
            'air_temperature_C': df_combined['AirTC_C_Avg'].tolist(),
            'air_temperature_D': df_combined['AirTC_D_Avg'].tolist(),
            'RH_A': df_combined['RH_A'].tolist(),
            'RH_B': df_combined['RH_B'].tolist(),
            'RH_C': df_combined['RH_C'].tolist(),
            'RH_D': df_combined['RH_D'].tolist(),
            'last_updated': datetime.now().isoformat(),
            'row_count': len(df_combined)
        }
        # Save to temporary file (same folder as .dat)
        json_path = dat_file_path.replace('.dat', '.json')
        with open(json_path, 'w') as f:
            json.dump(plot_data, f, indent=2,allow_nan=False)
        logger.info(f"JSON created at {json_path}")
        return json_path
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        return None


def push_to_github(source_json_path):
    """Copy JSON to local repo, commit, and force-push to a dedicated data-branch."""
    try:
        # 1. Copy the JSON into the repo folder
        dest_path = os.path.join(GITHUB_REPO_PATH, GITHUB_JSON_NAME)
        shutil.copy2(source_json_path, dest_path)
        logger.info(f"Copied JSON to {dest_path}")

        # 2. Run git commands
        original_dir = os.getcwd()
        os.chdir(GITHUB_REPO_PATH)

        # Attempt to switch to data-branch (use -f to force clean swap)
        checkout_attempt = subprocess.run(["git", "checkout", "-f", "data-branch"], capture_output=True)

        # If the branch doesn't exist, create and initialize it correctly
        if checkout_attempt.returncode != 0:
            logger.info("data-branch not found. Creating and initializing...")

            # Create the orphan branch
            subprocess.run(["git", "checkout", "--orphan", "data-branch"], check=True, capture_output=True)

            # Clear the index completely to start fresh
            subprocess.run(["git", "reset"], capture_output=True)

            # Create a tiny dummy commit to establish the branch
            with open(".gitkeep", "w") as f:
                f.write("data-branch placeholder")
            subprocess.run(["git", "add", ".gitkeep"], check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "Initial commit on data-branch"], check=True, capture_output=True)

        # Stage only the JSON file
        subprocess.run(["git", "add", GITHUB_JSON_NAME], check=True, capture_output=True)

        # Commit (Always amend the last commit to keep history at exactly 1 commit)
        commit_msg = f"Update measurements - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        # Amend the commit
        subprocess.run(["git", "commit", "--amend", "-m", commit_msg], check=True, capture_output=True)

        # Force push to update the branch instantly on GitHub
        subprocess.run(["git", "push", "origin", "data-branch", "--force"], check=True, capture_output=True)
        logger.info("Successfully force-pushed JSON to data-branch")

        # Safely switch back to the main branch (Using -f to force restore all website files)
        subprocess.run(["git", "checkout", "-f", "main"], check=True, capture_output=True)
        logger.info("Successfully restored main workspace")

        os.chdir(original_dir)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed: {e.stderr.decode()}")
        # Make sure we try to return to main even if something fails
        try:
            subprocess.run(["git", "checkout", "-f", "main"], capture_output=True)
            os.chdir(original_dir)
        except Exception:
            pass
        return False
    except Exception as e:
        logger.error(f"Error pushing to GitHub: {e}")
        return False

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_sync_time = 0
        self.debounce_seconds = DEBOUNCE_SECONDS
        self.pending_upload = False
        self.last_file_size = self.get_file_size()

    def get_file_size(self):
        if os.path.exists(LOCAL_DAT_FILE):
            return os.path.getsize(LOCAL_DAT_FILE)
        return 0

    def on_modified(self, event):
        if event.is_directory:
            return
        if not event.src_path.endswith(os.path.basename(LOCAL_DAT_FILE)):
            return
        current_size = self.get_file_size()
        if current_size == self.last_file_size:
            return
        self.last_file_size = current_size
        current_time = time.time()
        if current_time - self.last_sync_time < self.debounce_seconds:
            self.pending_upload = True
            return
        self.last_sync_time = current_time
        self.pending_upload = False
        logger.info(f"File changed - size: {current_size} bytes")
        time.sleep(1)
        self.do_sync()

    def process_pending(self):
        if self.pending_upload:
            current_time = time.time()
            if current_time - self.last_sync_time >= self.debounce_seconds:
                self.pending_upload = False
                self.last_file_size = self.get_file_size()
                logger.info("Processing pending upload")
                time.sleep(1)
                self.do_sync()

    def do_sync(self):
        json_path = convert_to_json(LOCAL_DAT_FILE,LOCAL_DAT_FILE2)
        if json_path and os.path.exists(json_path):
            success = push_to_github(json_path)
            # Remove temp JSON if you want (optional)
            # os.remove(json_path)
            if success:
                logger.info("Sync completed")
            else:
                logger.error("Sync failed")
        else:
            logger.error("Failed to create JSON")


def run_monitor():
    if not os.path.exists(LOCAL_DAT_FILE):
        logger.warning(f"Local file does not exist: {LOCAL_DAT_FILE}")
        logger.info("Waiting for file to be created...")

    # Initial sync (if file exists)
    if os.path.exists(LOCAL_DAT_FILE):
        logger.info("Performing initial sync...")
        handler = FileChangeHandler()
        handler.do_sync()
    else:
        handler = FileChangeHandler()

    watch_folder = os.path.dirname(LOCAL_DAT_FILE)
    if not watch_folder:
        watch_folder = '.'

    observer = Observer()
    observer.schedule(handler, path=watch_folder, recursive=False)
    observer.start()

    logger.info(f"Started monitoring: {LOCAL_DAT_FILE}")
    logger.info(f"Debounce: {DEBOUNCE_SECONDS} seconds")
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            handler.process_pending()
            # Also check for external changes
            if os.path.exists(LOCAL_DAT_FILE):
                current_size = os.path.getsize(LOCAL_DAT_FILE)
                if current_size != handler.last_file_size:
                    handler.last_file_size = current_size
                    logger.info(f"File changed externally - size: {current_size}")
                    time.sleep(1)
                    handler.do_sync()
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Stopping monitor...")
        observer.stop()
    observer.join()
    logger.info("Monitor stopped")


if __name__ == "__main__":
    run_monitor()