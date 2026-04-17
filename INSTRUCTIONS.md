# CrackSense Dashboard — Quick Start Guide

## 1. Local Access (Only for you)
1. Go to the folder: `D:\crack\CRACKAPP\CRACKAPP`
2. Double-click the file: **`run_dashboard.bat`**
3. Open your browser to: **http://localhost:8501**

## 2. Remote Access (HTTPS link for others)
If you want to share the dashboard with someone else:
1. Open a **PowerShell** or **Command Prompt**.
2. Run this command:
   ```powershell
   ssh -R 80:localhost:8501 nokey@localhost.run
   ```
3. Copy the **https://...** link that appears in the window and send it to them.

---
*Note: The HTTPS link will change every time you run the command.*
