# Windows Task Scheduler Setup for Daily Trading Review

## Quick Setup Guide

### Method 1: Using PowerShell Script (Recommended)

1. **Create PowerShell Setup Script**

Save as `setup_task_scheduler.ps1`:

```powershell
# Setup Task Scheduler for Daily Trading Review
# Run as Administrator

$ErrorActionPreference = "Stop"

Write-Host "=============================================="
Write-Host "Windows Task Scheduler Setup"
Write-Host "=============================================="
Write-Host ""

# Get project directory
$ProjectDir = Split-Path -Parent $PSScriptRoot
$PythonScript = Join-Path $ProjectDir "run_daily_review.py"

# Find Python executable
$PythonExe = $null
$VenvPython = Join-Path $ProjectDir "venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    $PythonExe = $VenvPython
    Write-Host "✓ Using virtual environment Python: $PythonExe"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonExe = (Get-Command python).Source
    Write-Host "✓ Using system Python: $PythonExe"
} else {
    Write-Host "❌ Python not found. Please install Python 3.8+"
    exit 1
}

# Verify script exists
if (-not (Test-Path $PythonScript)) {
    Write-Host "❌ run_daily_review.py not found at: $PythonScript"
    exit 1
}

# Create scheduled task
$TaskName = "MyTrader Daily Review"
$Description = "Daily paper trading performance review with AI insights"

# Check if task already exists
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($ExistingTask) {
    Write-Host "⚠️  Task already exists!"
    $Response = Read-Host "Replace existing task? (y/n)"
    if ($Response -ne "y") {
        Write-Host "Cancelled."
        exit 0
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "✓ Removed existing task"
}

# Create task action
$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "run_daily_review.py" `
    -WorkingDirectory $ProjectDir

# Create task trigger (daily at 6:00 PM)
$Trigger = New-ScheduledTaskTrigger -Daily -At 6:00PM

# Create task settings
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5)

# Create task principal (run whether user is logged on or not)
$Principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

# Register task
Register-ScheduledTask `
    -TaskName $TaskName `
    -Description $Description `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal

Write-Host ""
Write-Host "✓ Task scheduled successfully!"
Write-Host ""
Write-Host "Task Details:"
Write-Host "  Name: $TaskName"
Write-Host "  Schedule: Daily at 6:00 PM"
Write-Host "  Python: $PythonExe"
Write-Host "  Script: $PythonScript"
Write-Host "  Working Dir: $ProjectDir"
Write-Host ""
Write-Host "Verify in Task Scheduler:"
Write-Host "  taskschd.msc"
Write-Host ""
Write-Host "Test task manually:"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
Write-Host "View task history:"
Write-Host "  Get-ScheduledTaskInfo -TaskName '$TaskName'"
Write-Host ""
Write-Host "Remove task:"
Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
Write-Host "=============================================="
```

2. **Run PowerShell Script**

```powershell
# Right-click PowerShell and "Run as Administrator"
cd C:\path\to\MyTrader
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_task_scheduler.ps1
```

### Method 2: Manual Task Scheduler Setup

1. **Open Task Scheduler**
   - Press `Win + R`
   - Type `taskschd.msc`
   - Press Enter

2. **Create Basic Task**
   - Click "Create Basic Task..." in right panel
   - Name: `MyTrader Daily Review`
   - Description: `Daily paper trading performance review with AI insights`
   - Click "Next"

3. **Set Trigger**
   - Select "Daily"
   - Click "Next"
   - Start date: Today
   - Time: 18:00 (6:00 PM)
   - Recur every: 1 days
   - Click "Next"

4. **Set Action**
   - Select "Start a program"
   - Click "Next"
   - Program/script: Browse to Python executable
     - If using venv: `C:\path\to\MyTrader\venv\Scripts\python.exe`
     - Otherwise: `python.exe` or `C:\Python39\python.exe`
   - Add arguments: `run_daily_review.py`
   - Start in: `C:\path\to\MyTrader`
   - Click "Next"

5. **Finish and Configure**
   - Check "Open the Properties dialog..."
   - Click "Finish"

6. **Advanced Settings** (in Properties dialog)
   - General tab:
     - ☑ Run whether user is logged on or not
     - ☐ Run with highest privileges (not needed)
   - Triggers tab:
     - Edit trigger
     - ☑ Enabled
     - Advanced settings:
       - ☑ Repeat task every: (leave unchecked for once daily)
   - Actions tab:
     - Verify settings correct
   - Conditions tab:
     - ☐ Start the task only if the computer is on AC power (uncheck)
     - ☑ Start the task only if the computer is idle for: (leave unchecked)
   - Settings tab:
     - ☑ Allow task to be run on demand
     - ☑ Run task as soon as possible after a scheduled start is missed
     - If task fails, restart every: 5 minutes, 3 attempts
   - Click "OK"

7. **Test Task**
   - Right-click task → "Run"
   - Check logs: `C:\path\to\MyTrader\logs\`

## Verification

### Check Task Status

```powershell
# PowerShell
Get-ScheduledTask -TaskName "MyTrader Daily Review"
Get-ScheduledTaskInfo -TaskName "MyTrader Daily Review"
```

### Test Manual Execution

```powershell
# Run task immediately
Start-ScheduledTask -TaskName "MyTrader Daily Review"

# Check result
Get-ScheduledTaskInfo -TaskName "MyTrader Daily Review" | Select LastRunTime, LastTaskResult
```

### View Task History

1. Open Task Scheduler
2. Find "MyTrader Daily Review"
3. Click "History" tab (enable if disabled)
4. Review execution history

### Check Logs

```powershell
# View recent logs
Get-Content C:\path\to\MyTrader\logs\trading.log -Tail 100

# View cron-style logs (if configured)
Get-Content C:\path\to\MyTrader\logs\cron_review.log -Tail 100
```

## Troubleshooting

### Task Doesn't Run

**Check Task Status:**
```powershell
$Task = Get-ScheduledTask -TaskName "MyTrader Daily Review"
if ($Task.State -ne "Ready") {
    Write-Host "Task state: $($Task.State)"
}
```

**Common Issues:**
1. **Wrong Python path**: Verify executable exists
2. **Wrong working directory**: Must be project root
3. **Python not in PATH**: Use full path to python.exe
4. **Virtual environment**: Use venv\Scripts\python.exe

### Task Fails to Execute

**Check Last Result:**
```powershell
$Info = Get-ScheduledTaskInfo -TaskName "MyTrader Daily Review"
Write-Host "Last run: $($Info.LastRunTime)"
Write-Host "Result: $($Info.LastTaskResult)"
```

**Result Codes:**
- `0` (0x0): Success
- `1` (0x1): General error
- `267011` (0x41301): Task not run yet
- `2147943785` (0x800704C9): Task terminated

**Solutions:**
1. Test script manually:
   ```powershell
   cd C:\path\to\MyTrader
   python run_daily_review.py
   ```
2. Check Python installed and working
3. Verify config.yaml exists
4. Check AWS credentials configured
5. Review logs for errors

### Permission Issues

**Run as Different User:**
1. Task Properties → General
2. Click "Change User or Group..."
3. Enter your username
4. Enter password when prompted

**Run with Elevated Privileges:**
1. Task Properties → General
2. ☑ Run with highest privileges
3. Click "OK"

### Task Runs but No Output

**Enable Logging:**

Modify task action to redirect output:
- Program: `cmd.exe`
- Arguments: `/c python run_daily_review.py >> logs\scheduled_review.log 2>&1`
- Start in: `C:\path\to\MyTrader`

Or use Python logging (already configured in script).

## Custom Schedule Examples

### Weekdays Only at 5:00 PM

1. Edit trigger
2. Time: 17:00 (5:00 PM)
3. Recur every: 1 days
4. Advanced settings:
   - ☑ Days: Mon, Tue, Wed, Thu, Fri

### Multiple Times Per Day

Add multiple triggers:
1. Trigger 1: Daily at 12:00 (noon review)
2. Trigger 2: Daily at 18:00 (evening review)

### After System Startup

Add trigger:
- Type: "At startup"
- Delay: 5 minutes

## Monitoring

### Email Notifications (Optional)

Add to config.yaml:
```yaml
live_review:
  send_email_notifications: true
  email_recipients:
    - trader@example.com
  email_on_warnings_only: true
```

Requires email server configuration.

### Windows Event Log

Task execution logged to:
- Event Viewer → Windows Logs → Application
- Source: "Task Scheduler"

## Maintenance

### Update Schedule

```powershell
# Disable task
Disable-ScheduledTask -TaskName "MyTrader Daily Review"

# Re-enable
Enable-ScheduledTask -TaskName "MyTrader Daily Review"

# Modify trigger (GUI)
taskschd.msc  # Then edit manually
```

### Remove Task

```powershell
# PowerShell
Unregister-ScheduledTask -TaskName "MyTrader Daily Review" -Confirm:$false

# Or via GUI
# Task Scheduler → Right-click task → Delete
```

## Backup Task Configuration

### Export Task

```powershell
Export-ScheduledTask -TaskName "MyTrader Daily Review" | Out-File MyTrader_Task_Backup.xml
```

### Import Task

```powershell
Register-ScheduledTask -Xml (Get-Content MyTrader_Task_Backup.xml | Out-String) -TaskName "MyTrader Daily Review"
```

## Advanced Configuration

### Run Task on Multiple Conditions

```powershell
# Multiple triggers
$Trigger1 = New-ScheduledTaskTrigger -Daily -At 6:00PM
$Trigger2 = New-ScheduledTaskTrigger -Daily -At 9:00PM

Register-ScheduledTask `
    -TaskName "MyTrader Daily Review" `
    -Action $Action `
    -Trigger @($Trigger1, $Trigger2) `
    -Settings $Settings
```

### Run with Environment Variables

Create wrapper script `run_review_wrapper.ps1`:
```powershell
# Set environment variables
$env:AWS_PROFILE = "trading"
$env:PYTHONPATH = "C:\path\to\MyTrader"

# Run review
python run_daily_review.py
```

Schedule wrapper instead of Python script directly.

## Support

For issues:
1. Test manual execution first
2. Check logs: `logs\trading.log`
3. Verify Python and dependencies
4. Review Task Scheduler history
5. Check Event Viewer for errors

---

**Platform**: Windows 10/11  
**Task Scheduler Version**: 2.0+  
**Required Permissions**: Standard user (no admin needed)
