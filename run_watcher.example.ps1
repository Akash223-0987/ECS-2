# Example runner script for Google Sheets Priority Watcher
# Copy this file to run_watcher.ps1 and update with your own credentials and sheet ID.

$env:GOOGLE_APPLICATION_CREDENTIALS = "path\to\your\client_secret_or_service_account.json"
$env:GOOGLE_SHEET_ID = "your_google_sheet_id_here"
$env:GOOGLE_SHEET_WORKSHEET = "Sheet1"
$env:POLL_INTERVAL_SECONDS = "5.0"
$env:OAUTHLIB_RELAX_TOKEN_SCOPE = "1"

Write-Host "Starting Google Sheets Priority Watcher for Sheet ID: $env:GOOGLE_SHEET_ID" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the watcher at any time." -ForegroundColor Yellow
python -m ml_service.sheets_watcher
