# Demo launcher for the platform running on Docker Desktop Kubernetes.
#
# Checks the cluster, fixes the known "API pods started before MLflow" state,
# opens port-forwards, sends real predictions, opens the dashboard, and cleans
# the port-forwards up when you press Enter.
#
#   demo.bat                  the standard demo
#   demo.bat -Retrain         also run drift + retrain locally (the rejected-model moment)
#   demo.bat -Failure         also run the offline bad-candidate failure scenario
#   demo.bat -Traffic 200     send more prediction traffic (default 60)
#   demo.bat -NoBrowser       do not open the browser
#
# Read-only against the cluster except for one thing: if the inference API pods
# are not ready, it restarts that Deployment, which is the documented remedy.

param(
    [int]$Traffic = 60,
    [switch]$Retrain,
    [switch]$Failure,
    [switch]$NoBrowser,
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"
$Namespace = "ml-platform"
$Root = Split-Path -Parent $PSScriptRoot
$Forwards = @()

function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }
function Ok($text)   { Write-Host "    [ok] $text" -ForegroundColor Green }
function Warn($text) { Write-Host "    [!!] $text" -ForegroundColor Yellow }
function Fail($text) { Write-Host "    [xx] $text" -ForegroundColor Red }

function Test-PortInUse([int]$Port) {
    $listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return [bool]$listening
}

function Get-FreePort([int]$Preferred) {
    $port = $Preferred
    while (Test-PortInUse $port) { $port++ }
    return $port
}

function Start-Forward([string]$Service, [int]$LocalPort, [int]$RemotePort) {
    $proc = Start-Process kubectl `
        -ArgumentList "-n", $Namespace, "port-forward", "svc/$Service", "$($LocalPort):$RemotePort" `
        -WindowStyle Hidden -PassThru
    $script:Forwards += $proc
    return $proc
}

function Wait-Http([string]$Url, [int]$Seconds = 30) {
    for ($i = 0; $i -lt $Seconds; $i++) {
        try {
            $r = Invoke-WebRequest $Url -UseBasicParsing -TimeoutSec 3
            if ($r.StatusCode -eq 200) { return $true }
        } catch { }
        Start-Sleep -Seconds 1
    }
    return $false
}

$Payload = '{"applications":[{"term_months":84,"employees":12,"jobs_created":3,"jobs_retained":8,"gross_approved":250000,"sba_approved":187500,"disbursed":250000,"state":"CA","bank_state":"CA","revolving_line_of_credit":"N","low_doc":"N","urban_rural":1,"new_business":1,"naics":"722410","franchise_code":0,"approval_date":"2005-06-15","disbursement_date":"2005-07-20"}]}'

try {
    # --- 1. Docker and the cluster ------------------------------------------
    Step "Checking Docker and Kubernetes"
    docker version --format "{{.Server.Version}}" *> $null
    if ($LASTEXITCODE -ne 0) {
        Fail "Docker is not running. Start Docker Desktop, wait for Kubernetes to go green, and run this again."
        exit 1
    }
    Ok "Docker is running"

    kubectl get nodes --request-timeout=15s *> $null
    if ($LASTEXITCODE -ne 0) {
        Fail "Kubernetes is not reachable. Enable it in Docker Desktop settings, or wait for it to finish starting."
        exit 1
    }
    Ok "Kubernetes is reachable"

    # --- 2. Pods ------------------------------------------------------------
    Step "Checking pods in $Namespace"
    $pods = kubectl get pods -n $Namespace --no-headers --request-timeout=30s
    $pods | ForEach-Object { Write-Host "    $_" }

    $notReady = $pods | Where-Object { $_ -match '^inference-api' -and $_ -notmatch '\s1/1\s' }
    if ($notReady) {
        Warn "Inference API pods are not ready (they probably started before MLflow). Restarting them."
        kubectl -n $Namespace rollout status deploy/mlflow --timeout=180s | Out-Null
        kubectl -n $Namespace rollout restart deploy/inference-api | Out-Null
        kubectl -n $Namespace rollout status deploy/inference-api --timeout=240s
        if ($LASTEXITCODE -ne 0) { Fail "The API did not become ready. Check: kubectl -n $Namespace logs deploy/inference-api"; exit 1 }
        Ok "Inference API restarted and ready"
    } else {
        Ok "Inference API pods are ready"
    }

    # --- 3. Port-forwards ---------------------------------------------------
    Step "Opening port-forwards"
    $ApiPort = Get-FreePort 8080
    Start-Forward "inference-api" $ApiPort 80 | Out-Null
    $Api = "http://localhost:$ApiPort"
    if (-not (Wait-Http "$Api/health" 30)) { Fail "The API port-forward did not come up on $ApiPort"; exit 1 }
    Ok "API        $Api"

    # The dashboard's Grafana and Jaeger links point at these exact ports.
    foreach ($svc in @(@{Name="grafana"; Port=3000}, @{Name="jaeger"; Port=16686})) {
        if (Test-PortInUse $svc.Port) {
            Warn "$($svc.Name): port $($svc.Port) already in use, leaving it (an existing forward will still work)"
        } else {
            Start-Forward $svc.Name $svc.Port $svc.Port | Out-Null
            Ok "$($svc.Name.PadRight(10)) http://localhost:$($svc.Port)"
        }
    }

    # --- 4. Readiness and one prediction ------------------------------------
    Step "Readiness"
    $ready = Invoke-RestMethod "$Api/ready"
    Write-Host "    status: $($ready.status)  model: $($ready.model.name) v$($ready.model.version) @ $($ready.model.alias)"

    Step "One real prediction"
    $result = Invoke-RestMethod "$Api/predict" -Method Post -ContentType "application/json" -Body $Payload
    $p = $result.predictions[0]
    Write-Host "    default_probability: $($p.default_probability)  flagged: $($p.flagged)"
    Write-Host "    served_by: $($result.model.served_by)  tier: $($result.model.serving_tier)"

    # --- 5. Traffic ---------------------------------------------------------
    Step "Sending $Traffic predictions (fills the latency panels)"
    $okCount = 0; $failCount = 0
    for ($i = 1; $i -le $Traffic; $i++) {
        try {
            Invoke-RestMethod "$Api/predict" -Method Post -ContentType "application/json" `
                -Headers @{ "x-routing-key" = "demo-$i" } -Body $Payload -TimeoutSec 10 | Out-Null
            $okCount++
        } catch { $failCount++ }
        if ($i % 20 -eq 0) { Write-Host "    $i / $Traffic" }
        Start-Sleep -Milliseconds 500
    }
    Ok "$okCount succeeded, $failCount failed"

    # --- 6. Platform summary -------------------------------------------------
    Step "Platform summary (from /platform)"
    $health = Invoke-RestMethod "$Api/platform/health"
    Write-Host "    overall: $($health.status)   can_serve: $($health.can_serve)"
    foreach ($c in $health.components) { Write-Host ("      {0,-14} {1}" -f $c.name, $c.status) }
    $canary = Invoke-RestMethod "$Api/platform/canary"
    Write-Host "    canary traffic: $($canary.traffic_percent)%   active: $($canary.active)"
    $promo = Invoke-RestMethod "$Api/platform/promotions"
    Write-Host "    production version: $($promo.production_version)"

    # --- 7. Optional local demos ---------------------------------------------
    if ($Retrain) {
        Step "Drift check + retrain (local store; takes a few minutes)"
        Push-Location $Root
        try {
            uv run python -m ml_platform drift --scenario lending_shift
            Write-Host "    drift exit code: $LASTEXITCODE (2 = retrain recommended)"
            uv run python -m ml_platform retrain
        } finally { Pop-Location }
    }

    if ($Failure) {
        Step "Offline failure scenario: a bad candidate against the real gates"
        Push-Location $Root
        try { uv run python -m ml_platform failure --offline } finally { Pop-Location }
    }

    # --- 8. Open the dashboard ----------------------------------------------
    $Dashboard = "$Api/dashboard/"
    Step "Dashboard: $Dashboard"
    if (-not $NoBrowser) { Start-Process $Dashboard }

    if (-not $NoPause) {
        Write-Host ""
        Write-Host "Port-forwards are running. Press Enter to stop them and exit." -ForegroundColor Cyan
        [void](Read-Host)
    }
}
finally {
    if ($Forwards.Count -gt 0) {
        Step "Stopping port-forwards"
        foreach ($proc in $Forwards) {
            if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
        }
        Ok "stopped"
    }
}
