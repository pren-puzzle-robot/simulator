import requests
import time

GOPRO_IP = "10.5.5.9"

def send_command(path, desc=None):
    url = f"http://{GOPRO_IP}{path}"
    if desc:
        print(desc, "->", url)
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    return r

def main():
    # 1. Check API is reachable / wake camera
    try:
        print("Checking connection to GoPro...")
        requests.get(f"http://{GOPRO_IP}/gp/gpControl", timeout=5)
        print("GoPro reachable.")
    except Exception as e:
        print("Could not reach GoPro API:", e)
        return

    # 2. Set mode to Photo
    # For Hero 7 Black:
    #   /gp/gpControl/command/mode?p=1   -> Photo mode
    try:
        send_command("/gp/gpControl/command/mode?p=1", "Setting mode to PHOTO")
        time.sleep(1)
    except Exception as e:
        print("Failed to set mode:", e)
        return

    # 3. (Optional) Set photo sub-mode (single photo)
    # Many Hero 7 Black firmwares use:
    #   mode=1 (Photo), sub_mode=0 (Single)
    try:
        send_command("/gp/gpControl/command/sub_mode?mode=1&sub_mode=0",
                     "Setting sub-mode to SINGLE PHOTO")
        time.sleep(1)
    except Exception as e:
        print("Failed to set sub-mode (continuing anyway):", e)

    # 4. Trigger shutter (take photo)
    try:
        send_command("/gp/gpControl/command/shutter?p=1", "Triggering shutter (take photo)")
        # For single photo you can optionally send p=0 after a short delay
        time.sleep(2)
        send_command("/gp/gpControl/command/shutter?p=0", "Stopping shutter")
        print("Photo command sent successfully.")
    except Exception as e:
        print("Failed to trigger shutter:", e)

if __name__ == "__main__":
    main()
