<#
.SYNOPSIS
    Automated deployment script for MIDOG Detector (KongNet_Det_MIDOG_1 / YOLO) to Google Cloud Vertex AI.

.DESCRIPTION
    1. Validates selected model engine (KongNet winner or YOLO).
    2. Ensures required GCP services are enabled (Artifact Registry, Cloud Build, Vertex AI).
    3. Creates Google Artifact Registry repository (oncogemma-models) if not existing.
    4. Submits container build using Google Cloud Build.
    5. Registers the container in Vertex AI Model Registry.
    6. Creates a Vertex AI Endpoint.
    7. Deploys the model to the Endpoint (T4 GPU by default, an L4, or CPU).
    8. Prints the Endpoint ID and the OncoGemma .env configuration line.

    Throughput (OncoGemma v7.2 Stage 3 spec §12.1): -Gpu l4 serves on g2-standard-4 + L4; -MaxReplicas above
    -MinReplicas autoscales on GPU duty cycle (-DutyCycleTarget); -MaxBatch sets the patches per forward pass;
    -Precision fp16 runs under autocast and the container refuses to start unless the golden tiles match the fp32
    record on that GPU. Measure G3-3 before and after each change; the calibration card is fitted on fp32 logits.

.EXAMPLE
    .\deploy.ps1 -ProjectId "oncogemma" -Region "us-central1" -DeployGPU
    .\deploy.ps1 -Gpu l4 -MinReplicas 1 -MaxReplicas 3 -MaxBatch 32
    .\deploy.ps1 -DeployCPU
#>

param(
    [string]$ProjectId = "oncogemma",
    [string]$Region = "us-central1",
    [string]$RepoName = "oncogemma-repo",
    [string]$ImageTag = "midog-detector:v3.1",
    [string]$EndpointName = "mitosis-detector-endpoint",
    [string]$ModelDisplayName = "midog-kongnet-v3-1",
    [switch]$DeployGPU,
    [switch]$DeployCPU,
    [ValidateSet("t4", "l4")]
    [string]$Gpu = "t4",
    [ValidateRange(1, 10)]
    [int]$MinReplicas = 1,
    [ValidateRange(1, 10)]
    [int]$MaxReplicas = 1,
    [ValidateRange(10, 100)]
    [int]$DutyCycleTarget = 60,
    [ValidateRange(1, 64)]
    [int]$MaxBatch = 16,
    [ValidateSet("fp32", "fp16")]
    [string]$Precision = "fp32",
    # The container's runtime identity: it reads tile packs from GCS (tile_pack_uri). OncoGemma's Terraform creates
    # it and grants it read on the tile-pack buckets (infra/terraform/iam.tf, storage.tf). Without it the model runs
    # as a Google-managed identity that bucket IAM cannot reach, and every tile pack answers pack_read_error 403.
    [string]$ServiceAccount = "midog-endpoint-sa@$ProjectId.iam.gserviceaccount.com"
)

# Default to GPU if neither or both specified
$useGpu = $true
if ($DeployCPU -and -not $DeployGPU) {
    $useGpu = $false
}
if ($MaxReplicas -lt $MinReplicas) {
    Write-Host "[ERROR] -MaxReplicas ($MaxReplicas) is below -MinReplicas ($MinReplicas)." -ForegroundColor Red
    exit 1
}
if ($Precision -eq "fp16" -and -not $useGpu) {
    Write-Host "[ERROR] -Precision fp16 needs a GPU." -ForegroundColor Red
    exit 1
}
$gpuSpec = @{
    "t4" = @{ Machine = "n1-standard-4"; Accelerator = "type=nvidia-tesla-t4,count=1"; Label = "NVIDIA T4 GPU (n1-standard-4)" }
    "l4" = @{ Machine = "g2-standard-4"; Accelerator = "type=nvidia-l4,count=1"; Label = "NVIDIA L4 GPU (g2-standard-4)" }
}[$Gpu]

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  MIDOG Mitosis Detector Vertex AI Deployment Pipeline (v2)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "GCP Project : $ProjectId"
Write-Host "Region      : $Region"
Write-Host "Repository  : $RepoName"
Write-Host "Image Tag   : $ImageTag"
Write-Host "Model Engine: KongNet_Det_MIDOG_1 (1st Place Challenge Winner)"
Write-Host "Hardware    : $(if ($useGpu) {$gpuSpec.Label} else {'CPU Only (e2-standard-4)'})"
Write-Host "Replicas    : $MinReplicas-$MaxReplicas$(if ($MaxReplicas -gt $MinReplicas) {" (GPU duty cycle target $DutyCycleTarget%)"})"
Write-Host "Batch       : $MaxBatch patches per forward pass, $Precision"
Write-Host "=========================================================="

# 1. Verify model engine
Write-Host "[INFO] Deploying KongNet_Det_MIDOG_1 (Official MIDOG Challenge Winner, v2 contract)." -ForegroundColor Green
Write-Host "[INFO] Model weights will be pre-cached into container during Cloud Build." -ForegroundColor Gray

# 2. Configure GCP Project
Write-Host "`n[Step 1/6] Configuring gcloud project..." -ForegroundColor Green
gcloud config set project $ProjectId

# 3. Enable Required Google Cloud APIs
Write-Host "`n[Step 2/6] Ensuring required Google Cloud APIs are enabled..." -ForegroundColor Green
gcloud services enable `
    artifactregistry.googleapis.com `
    cloudbuild.googleapis.com `
    aiplatform.googleapis.com

# 4. Create Artifact Registry repository if not already existing
Write-Host "`n[Step 3/6] Checking Artifact Registry repository '$RepoName'..." -ForegroundColor Green
$existingRepo = gcloud artifacts repositories describe $RepoName --location=$Region --format="value(name)" 2>$null
if (-not $existingRepo) {
    Write-Host "Creating Artifact Registry repository '$RepoName'..." -ForegroundColor Yellow
    gcloud artifacts repositories create $RepoName `
        --repository-format=docker `
        --location=$Region `
        --description="OncoGemma Model Containers"
} else {
    Write-Host "Repository '$RepoName' already exists." -ForegroundColor Gray
}

# 5. Build and Push Container using Cloud Build
$IMAGE_URI = "${Region}-docker.pkg.dev/${ProjectId}/${RepoName}/${ImageTag}"
Write-Host "`n[Step 4/6] Building and pushing container via Google Cloud Build..." -ForegroundColor Green
Write-Host "Target Image: $IMAGE_URI" -ForegroundColor Gray
gcloud builds submit --tag $IMAGE_URI

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Cloud Build failed! Please review the build logs above." -ForegroundColor Red
    exit 1
}

# 6. Upload Model to Vertex AI Model Registry
Write-Host "`n[Step 5/6] Registering model in Vertex AI Model Registry..." -ForegroundColor Green
$uploadOutput = gcloud ai models upload `
    --region=$Region `
    --display-name=$ModelDisplayName `
    --container-image-uri=$IMAGE_URI `
    --container-predict-route="/predict" `
    --container-health-route="/health" `
    --container-ports=8080 `
    --container-env-vars="MIDOG_MAX_BATCH=$MaxBatch,MIDOG_PRECISION=$Precision" `
    --format="value(model)"

Write-Host "Model uploaded successfully: $uploadOutput" -ForegroundColor Cyan

# Extract model ID or fetch latest model ID matching display name
$MODEL_ID = gcloud ai models list `
    --region=$Region `
    --filter="displayName:$ModelDisplayName" `
    --sort-by="~createTime" `
    --limit=1 `
    --format="value(name)"

Write-Host "Active Model Resource: $MODEL_ID" -ForegroundColor Gray

# 7. Create or Retrieve Vertex AI Endpoint
Write-Host "`n[Step 6/6] Creating or retrieving Vertex AI Endpoint '$EndpointName'..." -ForegroundColor Green
$ENDPOINT_ID = gcloud ai endpoints list `
    --region=$Region `
    --filter="displayName:$EndpointName" `
    --limit=1 `
    --format="value(name)"

if (-not $ENDPOINT_ID) {
    Write-Host "Creating new Endpoint '$EndpointName'..." -ForegroundColor Yellow
    gcloud ai endpoints create `
        --region=$Region `
        --display-name=$EndpointName

    $ENDPOINT_ID = gcloud ai endpoints list `
        --region=$Region `
        --filter="displayName:$EndpointName" `
        --limit=1 `
        --format="value(name)"
}

Write-Host "Vertex Endpoint Resource ID: $ENDPOINT_ID" -ForegroundColor Cyan

# Check for and undeploy existing models on the endpoint to avoid GPU quota conflicts
$deployedModels = gcloud ai endpoints describe $ENDPOINT_ID --region=$Region --format="value(deployedModels[].id)"
if ($deployedModels) {
    foreach ($dId in ($deployedModels -split '\r?\n')) {
        $dIdClean = $dId.Trim()
        if ($dIdClean) {
            Write-Host "Undeploying previous deployed model '$dIdClean' from endpoint to release GPU quota..." -ForegroundColor Yellow
            gcloud ai endpoints undeploy-model $ENDPOINT_ID --region=$Region --deployed-model-id=$dIdClean --quiet
        }
    }
}

# 8. Deploy Model to the Endpoint
Write-Host "`nDeploying model to Endpoint..." -ForegroundColor Green
if ($useGpu) {
    Write-Host "Deploying with $($gpuSpec.Label), $MinReplicas-$MaxReplicas replicas..." -ForegroundColor Yellow
    $deployArgs = @(
        "ai", "endpoints", "deploy-model", $ENDPOINT_ID,
        "--region=$Region",
        "--model=$MODEL_ID",
        "--display-name=${ModelDisplayName}-$Gpu-deploy",
        "--machine-type=$($gpuSpec.Machine)",
        "--accelerator=$($gpuSpec.Accelerator)",
        "--min-replica-count=$MinReplicas",
        "--max-replica-count=$MaxReplicas",
        "--service-account=$ServiceAccount"
    )
    if ($MaxReplicas -gt $MinReplicas) {
        $deployArgs += "--autoscaling-metric-specs=aiplatform.googleapis.com/prediction/online/accelerator/duty_cycle=$DutyCycleTarget"
    }
    & gcloud @deployArgs
} else {
    Write-Host "Deploying on CPU (e2-standard-4)..." -ForegroundColor Yellow
    gcloud ai endpoints deploy-model $ENDPOINT_ID `
        --region=$Region `
        --model=$MODEL_ID `
        --display-name="${ModelDisplayName}-cpu-deploy" `
        --machine-type="e2-standard-4" `
        --min-replica-count=1 `
        --max-replica-count=1 `
        --service-account=$ServiceAccount
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n==========================================================" -ForegroundColor Green
    Write-Host "  DEPLOYMENT COMPLETE!" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
    $numericId = ($ENDPOINT_ID.ToString().Trim() -split '/')[-1]
    Write-Host "Endpoint Resource: $ENDPOINT_ID" -ForegroundColor Cyan
    Write-Host "Endpoint ID      : $numericId" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Add this to your OncoGemma backend/.env file:" -ForegroundColor Yellow
    Write-Host "MIDOG_VERTEX_ENDPOINT_ID=$numericId" -ForegroundColor White
    Write-Host "VERTEX_PROJECT_ID=$ProjectId" -ForegroundColor White
    Write-Host "VERTEX_REGION=$Region" -ForegroundColor White
    Write-Host "=========================================================="
} else {
    Write-Host "`n[ERROR] Model deployment to endpoint failed." -ForegroundColor Red
}
