# Clear the contents of the .env file
Set-Content -Path .env -Value ""

# Append new values to the .env file
$azureTenantId = azd env get-value AZURE_TENANT_ID
$azureOpenAiEndpoint = azd env get-value AZURE_OPENAI_ENDPOINT
$azureOpenAiEmbeddingDeployment = azd env get-value AZURE_OPENAI_EMBEDDING_DEPLOYMENT
$azureOpenAiEmbeddingModel = azd env get-value AZURE_OPENAI_EMBEDDING_MODEL
$azureOpenAiEmbeddingDimensions = azd env get-value AZURE_OPENAI_EMBEDDING_DIMENSIONS
$azureAiVisionEndpoint = azd env get-value AZURE_AI_VISION_ENDPOINT

Add-Content -Path .env -Value "API_HOST=azure"
Add-Content -Path .env -Value "AZURE_TENANT_ID=$azureTenantId"
Add-Content -Path .env -Value "AZURE_OPENAI_ENDPOINT=$azureOpenAiEndpoint"
Add-Content -Path .env -Value "AZURE_OPENAI_VERSION=2024-10-21"
Add-Content -Path .env -Value "AZURE_OPENAI_EMBEDDING_DEPLOYMENT=$azureOpenAiEmbeddingDeployment"
Add-Content -Path .env -Value "AZURE_OPENAI_EMBEDDING_MODEL=$azureOpenAiEmbeddingModel"
Add-Content -Path .env -Value "AZURE_OPENAI_EMBEDDING_DIMENSIONS=$azureOpenAiEmbeddingDimensions"
Add-Content -Path .env -Value "AZURE_AI_VISION_ENDPOINT=$azureAiVisionEndpoint"
