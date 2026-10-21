import os
import sys
from pathlib import Path
import requests
from dotenv import load_dotenv

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load .env if present
load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

def mask_text(text: object) -> str:
    s = str(text)
    if TOKEN:
        s = s.replace(TOKEN, "[REDACTED_TOKEN]")
    return s

def main():
    print("=" * 50)
    print("TELEGRAM CONFIGURATION & CONNECTIVITY TEST")
    print("=" * 50)

    if not TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN is not set!")
        print("Please configure it in a .env file or set it in your PowerShell terminal:")
        print('  $env:TELEGRAM_BOT_TOKEN="your_token_here"')
        print('  $env:TELEGRAM_CHAT_ID="your_chat_id_here"')
        sys.exit(1)

    if not CHAT_ID:
        print("ERROR: TELEGRAM_CHAT_ID is not set!")
        print("Please configure it in .env or via $env:TELEGRAM_CHAT_ID.")
        sys.exit(1)

    print(f"Token status: Configured (length={len(TOKEN)})")
    print(f"Chat ID: {CHAT_ID}")

    base_url = f"https://api.telegram.org/bot{TOKEN}"

    # 1. DNS & Connectivity test to Telegram API
    print("\n[1/3] Testing connection to Telegram API (getMe)...")
    try:
        response = requests.get(f"{base_url}/getMe", timeout=15)
        data = response.json()
        if response.ok and data.get("ok"):
            bot_user = data["result"].get("username", "Unknown")
            bot_name = data["result"].get("first_name", "Unknown")
            print(f"  SUCCESS: Connected to bot @{bot_user} ({bot_name})")
        else:
            print("  ERROR: Telegram API returned failure:", mask_text(data))
            sys.exit(1)
    except requests.RequestException as exc:
        print("  NETWORK ERROR during getMe:", mask_text(exc))
        sys.exit(1)

    # 2. Send test text message
    print("\n[2/3] Sending test text message...")
    try:
        response = requests.post(
            f"{base_url}/sendMessage",
            data={
                "chat_id": CHAT_ID,
                "text": "Traffic Monitoring System connected successfully!"
            },
            timeout=15,
        )
        data = response.json()
        if response.ok and data.get("ok"):
            print("  SUCCESS: Test message sent successfully!")
        else:
            print("  ERROR: Failed to send message:", mask_text(data))
            sys.exit(1)
    except requests.RequestException as exc:
        print("  NETWORK ERROR during sendMessage:", mask_text(exc))
        sys.exit(1)

    # 3. Find and send latest evidence photo
    print("\n[3/3] Sending latest evidence photo...")
    evidence_dir = Path(__file__).parent / "evidence"
    if not evidence_dir.exists():
        print("  Evidence directory does not exist. Skipping photo test.")
        return

    photos = sorted(
        [
            p for p in evidence_dir.glob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not photos:
        print("  No evidence photos found in evidence folder. Text message test succeeded.")
        return

    photo = photos[0]
    print(f"  Found latest evidence photo: {photo.name} ({photo.stat().st_size} bytes)")

    try:
        with photo.open("rb") as image_file:
            response = requests.post(
                f"{base_url}/sendPhoto",
                data={
                    "chat_id": CHAT_ID,
                    "caption": f"Traffic evidence test\n{photo.name}",
                },
                files={"photo": image_file},
                timeout=30,
            )
        data = response.json()
        if response.ok and data.get("ok"):
            print(f"  SUCCESS: Evidence photo {photo.name} sent to Telegram!")
        else:
            print("  ERROR: Telegram returned error for photo:", mask_text(data))
    except requests.RequestException as exc:
        print("  NETWORK ERROR during sendPhoto:", mask_text(exc))

if __name__ == "__main__":
    main()
