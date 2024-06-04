import configparser
from io import BytesIO
from azure.storage.blob import BlobServiceClient

config = configparser.ConfigParser()
config.read('config.ini')

# Azure Storage Account details
azure_storage_account_name = config['StorageAzure']['account_name']
azure_storage_account_key = config['StorageAzure']['account_key']
container_name = config['StorageAzure']['container_name']

# Function to upload file to Azure Storage
def upload_to_azure_storage(file):
    blob_service_client = BlobServiceClient.from_connection_string(f"DefaultEndpointsProtocol=https;AccountName={azure_storage_account_name};AccountKey={azure_storage_account_key}")
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=file.name)
    blob_client.upload_blob(file)