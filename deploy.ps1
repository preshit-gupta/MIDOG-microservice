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
    7. Deploys the model to the Endpoint (T4 GPU by default, or CPU).
    8. Prints the Endpoint ID and the OncoGemma .env configuration line.

.EXAMPLE
    .\deploy.ps1 -ProjectId "oncogemma" -Region "us-central1" -DeployGPU
    .\deploy.ps1 -DeployCPU
#>

param(
    [string]$ProjectId = "oncogemma",
    [string]$Region = "us-central1",
    [string]$RepoName = "oncogemma-repo",
    [string]$ImageTag = "midog-detector:v1",
    [string]$EndpointName = "mitosis-detector-endpoint",
    [string]$ModelDisplayName = "midog-kongnet-v1",
    [string]$Engine = "kongnet",
    [switch]$DeployGPU,
    [switch]$DeployCPU
)

# Default to GPU if neither or both specified
$useGpu = $true
if ($DeployCPU -and -not $DeployGPU) {
    $useGpu = $false
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  MIDOG Mitosis Detector Vertex AI Deployment Pipeline" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "GCP Project : $ProjectId"
Write-Host "Region      : $Region"
Write-Host "Repository  : $RepoName"
Write-Host "Image Tag   : $ImageTag"
Write-Host "Model Engine: $(if ($Engine -eq 'kongnet') {'KongNet_Det_MIDOG_1 (1st Place Challenge Winner)'} else {'YOLO (' + $Engine + ')'})"
Write-Host "Hardware    : $(if ($useGpu) {'NVIDIA T4 GPU (n1-standard-4)'} else {'CPU Only (e2-standard-4)'})"
Write-Host "=========================================================="

# 1. Verify model engine & weights
if ($Engine -eq "yolo") {
    if (-not (Test-Path "best.pt")) {
        Write-Host "[ERROR] 'best.pt' not found in current directory for YOLO engine!" -ForegroundColor Red
        Write-Host "Please place your MIDOG model checkpoint as 'best.pt' or deploy with -Engine kongnet" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "[INFO] Deploying KongNet_Det_MIDOG_1 (Official MIDOG Challenge Winner)." -ForegroundColor Green
    Write-Host "[INFO] Model weights will be pre-cached into container during Cloud Build." -ForegroundColor Gray
}

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
    Write-Host "Deploying with 1x NVIDIA Tesla T4 GPU on n1-standard-4..." -ForegroundColor Yellow
    gcloud ai endpoints deploy-model $ENDPOINT_ID `
        --region=$Region `
        --model=$MODEL_ID `
        --display-name="${ModelDisplayName}-gpu-deploy" `
        --machine-type="n1-standard-4" `
        --accelerator="type=nvidia-tesla-t4,count=1" `
        --min-replica-count=1 `
        --max-replica-count=1
} else {
    Write-Host "Deploying on CPU (e2-standard-4)..." -ForegroundColor Yellow
    gcloud ai endpoints deploy-model $ENDPOINT_ID `
        --region=$Region `
        --model=$MODEL_ID `
        --display-name="${ModelDisplayName}-cpu-deploy" `
        --machine-type="e2-standard-4" `
        --min-replica-count=1 `
        --max-replica-count=1
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
