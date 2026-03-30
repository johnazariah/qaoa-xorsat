// QAOA-XORSAT Azure Container Instances deployment
// Deploys 15 ACI instances (one per (k,D) pair) with shared blob storage for results.
//
// Usage:
//   az deployment group create -g qaoa-xorsat -f infra/main.bicep \
//     -p acrName=<your-acr> pMax=13 imageTag=latest

@description('Name of the ACR containing the qaoa-xorsat image')
param acrName string

@description('ACR image tag')
param imageTag string = 'latest'

@description('Maximum QAOA depth to compute')
@minValue(8)
@maxValue(15)
param pMax int = 13

@description('Azure region')
param location string = resourceGroup().location

@description('Number of CPU cores per container')
param cpuCores int = 16

@description('Memory in GB per container — must match pMax requirements')
param memoryGB int = 128

// Storage account for results
var storageAccountName = 'qaoa${uniqueString(resourceGroup().id)}'
var shareName = 'results'

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
  }
}

resource fileService 'Microsoft.Storage/storageAccounts/fileServices@2023-01-01' = {
  parent: storageAccount
  name: 'default'
}

resource fileShare 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-01-01' = {
  parent: fileService
  name: shareName
  properties: {
    shareQuota: 10 // GB — results are tiny
  }
}

// The 15 (k,D) pairs from Jordan et al.
var pairs = [
  { k: 3, d: 4 }
  { k: 3, d: 5 }
  { k: 3, d: 6 }
  { k: 3, d: 7 }
  { k: 3, d: 8 }
  { k: 4, d: 5 }
  { k: 4, d: 6 }
  { k: 4, d: 7 }
  { k: 4, d: 8 }
  { k: 5, d: 6 }
  { k: 5, d: 7 }
  { k: 5, d: 8 }
  { k: 6, d: 7 }
  { k: 6, d: 8 }
  { k: 7, d: 8 }
]

// Deploy one ACI per (k,D) pair
resource containerGroups 'Microsoft.ContainerInstance/containerGroups@2023-05-01' = [for (pair, i) in pairs: {
  name: 'qaoa-k${pair.k}-d${pair.d}'
  location: location
  properties: {
    osType: 'Linux'
    restartPolicy: 'Never'
    imageRegistryCredentials: [
      {
        server: '${acrName}.azurecr.io'
        #disable-next-line use-resource-id-functions
        username: acrName
        password: listKeys(resourceId('Microsoft.ContainerRegistry/registries', acrName), '2023-07-01').passwords[0].value
      }
    ]
    containers: [
      {
        name: 'qaoa'
        properties: {
          image: '${acrName}.azurecr.io/qaoa-xorsat:${imageTag}'
          command: [
            'julia'
            '--project=.'
            '-t'
            'auto'
            'scripts/optimize_qaoa.jl'
            '${pair.k}'
            '${pair.d}'
            '1'
            '${pMax}'
            '2'
            '320'
            '1234'
            'true'
            'adjoint'
          ]
          resources: {
            requests: {
              cpu: cpuCores
              memoryInGB: memoryGB
            }
          }
          volumeMounts: [
            {
              name: 'results'
              mountPath: '/workspace/results'
            }
          ]
        }
      }
    ]
    volumes: [
      {
        name: 'results'
        azureFile: {
          shareName: shareName
          storageAccountName: storageAccount.name
          storageAccountKey: storageAccount.listKeys().keys[0].value
        }
      }
    ]
  }
}]

// Outputs
output storageAccountName string = storageAccount.name
output shareName string = shareName
output containerNames array = [for (pair, i) in pairs: 'qaoa-k${pair.k}-d${pair.d}']
