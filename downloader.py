import os
import json
import time
import datetime
import schedule
import copernicusmarine

# --- Configuration ---
DATA_DIR = "./data"
TRACKING_FILE = "downloaded_files.json"

# Replace with your specific Copernicus Dataset ID
# The new Copernicus Marine Data Store uses dataset IDs (not product IDs).
# For Global Ocean OSTIA Sea Surface Temperature, the daily dataset ID is:
DATASET_ID = "cmems_mod_glo_phy_anfc_0.083deg-sst-anomaly_P1D-m_202411" 

# Copernicus Credentials (best practice is to use environment variables)
USERNAME = os.environ.get("COP_USER", "rdimitrov1")
PASSWORD = os.environ.get("COP_PASS", "#2VsL5hgDvG.XZL")

# Bounding box for your region of interest (Optional, but saves space)
# Example: Black Sea region (approximate)
# MIN_LON, MAX_LON = 27.0, 42.0
# MIN_LAT, MAX_LAT = 40.0, 47.0

def load_tracking():
    """Loads the list of already downloaded dates from a JSON file."""
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, 'r') as f:
            return json.load(f)
    return []

def save_tracking(downloaded_dates):
    """Saves the list of downloaded dates to a JSON file."""
    with open(TRACKING_FILE, 'w') as f:
        json.dump(downloaded_dates, f, indent=4)

def download_daily_data():
    """Checks for missing days and downloads them from Copernicus."""
    print(f"\\n[{datetime.datetime.now()}] Starting daily check for new NetCDF files...")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    downloaded = load_tracking()
    
    # Start checking from Jan 1, 2023 up to today
    start_date = datetime.date(2023, 1, 1)
    today = datetime.date.today()
    
    current_date = start_date
    new_downloads = 0
    
    while current_date <= today:
        date_str = current_date.strftime("%Y-%m-%d")
        
        if date_str not in downloaded:
            print(f"[*] Missing data for {date_str}. Attempting to download...")
            
            output_filename = f"sst_{date_str}.nc"
            date_filter = f"*{date_str.replace('-', '')}*"
            
            success = False
            last_error = None
            
            # Try both the new and old dataset IDs
            dataset_ids_to_try = [
                DATASET_ID,
                DATASET_ID.replace("_202411", "")
            ]
            
            for did in dataset_ids_to_try:
                try:
                    copernicusmarine.get(
                        dataset_id=did,
                        username=USERNAME,
                        password=PASSWORD,
                        filter=date_filter,
                        output_directory=DATA_DIR,
                        force_download=True,
                        no_directories=True,
                    )
                    
                    # Find the downloaded file and rename it
                    import glob
                    downloaded_files = glob.glob(os.path.join(DATA_DIR, f"*{date_str.replace('-', '')}*.nc"))
                    if downloaded_files:
                        # Rename the first matching file to our standard format
                        os.rename(downloaded_files[0], os.path.join(DATA_DIR, output_filename))
                        success = True
                        break # Success, stop trying other dataset IDs
                    else:
                        last_error = Exception(f"File for {date_str} was not found after download.")
                        
                except Exception as e:
                    last_error = e
                    
            if success:
                # If successful, add to tracking file
                downloaded.append(date_str)
                save_tracking(downloaded)
                new_downloads += 1
                print(f"[+] Successfully downloaded {date_str}")
            else:
                error_msg = str(last_error).lower()
                if "credentials" in error_msg or "username" in error_msg or "password" in error_msg or "401" in error_msg:
                    print(f"\\n❌ CRITICAL ERROR: Authentication failed. Please check your Copernicus username and password.")
                    print(f"Error details: {last_error}")
                    print("Stopping download process to prevent spamming the server.")
                    break # Stop the entire loop immediately
                
                print(f"[-] Failed to download {date_str}. Error: {last_error}")
                # We break here so we don't spam the server if today's data isn't ready yet
                if current_date == today:
                    break
        
        current_date += datetime.timedelta(days=1)
        
    print(f"[{datetime.datetime.now()}] Check complete. {new_downloads} new files downloaded.")

def start_scheduler():
    """Runs the downloader immediately, then schedules it to run every day."""
    # Run once immediately on startup
    download_daily_data()
    
    # Schedule to run every day at 14:00 (adjust time as needed based on when Copernicus updates)
    schedule.every().day.at("14:00").do(download_daily_data)
    
    print("\\nâ° Scheduler started. Waiting for the next scheduled run...")
    while True:
        schedule.run_pending()
        time.sleep(60) # Check schedule every minute

if __name__ == "__main__":
    if USERNAME == "your_copernicus_username":
        print("âš ï¸  WARNING: Please set your Copernicus username and password in the script or via environment variables (COP_USER, COP_PASS).")
    
    start_scheduler()
