# Upload data depuis ton PC Windows vers Vast.ai
# Usage : .\upload_data.ps1 -SshConnection "ssh -p XXXXX root@XX.XX.XX.XX"
# Le SSH connection string est donne par Vast.ai dans l'interface

param(
    [Parameter(Mandatory=$true)]
    [string]$SshConnection
)

# Extrait port et IP depuis "ssh -p XXXXX root@XX.XX.XX.XX"
if ($SshConnection -match "ssh -p (\d+) root@([\d\.]+)") {
    $port = $matches[1]
    $ip = $matches[2]
    Write-Host "SSH Port: $port"
    Write-Host "SSH IP  : $ip"
} else {
    Write-Host "ERREUR : Format SSH attendu : 'ssh -p XXXXX root@XX.XX.XX.XX'"
    exit 1
}

# 1. Upload caches Admiral (XAUUSD, XAGUSD, DXY + autres deja en cache)
Write-Host ""
Write-Host "[1/2] Upload data/cache/ (~390 MB)..."
scp -P $port -r "c:\Users\Shadow\TradingBot\data\cache" "root@${ip}:/workspace/TradingBot/data/"

# 2. Upload data_admiral (XAUUSD/XAGUSD/DXY parquet)
Write-Host ""
Write-Host "[2/2] Upload data_admiral/ (~164 MB)..."
scp -P $port -r "c:\Users\Shadow\TradingBot\data_admiral" "root@${ip}:/workspace/TradingBot/"

Write-Host ""
Write-Host "=== UPLOAD TERMINE ==="
Write-Host ""
Write-Host "Prochaine etape - connecte-toi au serveur :"
Write-Host "  $SshConnection"
Write-Host ""
Write-Host "Puis lance le build :"
Write-Host "  cd /workspace/TradingBot && bash vastai_setup/build_all.sh"
