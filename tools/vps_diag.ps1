# Diagnostic VPS : RAM, CPU, derniers crashs Python
$os = Get-CimInstance Win32_OperatingSystem
$totalGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
$freeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
Write-Output "RAM totale : $totalGB GB"
Write-Output "RAM libre  : $freeGB GB"
$cs = Get-CimInstance Win32_ComputerSystem
Write-Output "CPU cores  : $($cs.NumberOfLogicalProcessors)"
Write-Output "--- Derniers crashs (Application Error) ---"
try {
    Get-EventLog -LogName Application -EntryType Error -Newest 8 -ErrorAction Stop |
        Where-Object { $_.Source -match "Application Error|Python|.NET Runtime" } |
        ForEach-Object {
            Write-Output ("[" + $_.TimeGenerated + "] " + $_.Source)
            $m = $_.Message
            if ($m.Length -gt 300) { $m = $m.Substring(0, 300) }
            Write-Output $m
            Write-Output "---"
        }
} catch {
    Write-Output "Pas d'acces EventLog : $_"
}
