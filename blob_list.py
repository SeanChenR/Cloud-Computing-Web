import re
import configparser
from azure.storage.blob import BlobServiceClient

config = configparser.ConfigParser()
config.read('config.ini')

# Azure Storage Account details
azure_storage_account_name = config['StorageAzure']['account_name']
azure_storage_account_key = config['StorageAzure']['account_key']
container_name = config['StorageAzure']['container_name']

def get_blob_list(filename_ex):
    blob_service_client = BlobServiceClient.from_connection_string(f"DefaultEndpointsProtocol=https;AccountName={azure_storage_account_name};AccountKey={azure_storage_account_key}")
    blob_client = blob_service_client.get_container_client(container=container_name)
    blobs = blob_client.list_blobs()
    if filename_ex == "image":
        pattern = r'\b\w+\.(jpg|png)\b'
    elif filename_ex == "pdf":
        pattern = r'\b\w+\.(pdf)\b'
    elif filename_ex == "txt":
        pattern = r'\b\w+\.(txt)\b'
    else:
        raise ValueError("Invalid filename extension, Please use the file extension with [jpg, png, pdf, txt]")
    return [blob.name for blob in blobs if re.search(pattern, str(blob.name))]

if __name__ == '__main__':
    blob_list = get_blob_list("pdf")
    print(blob_list)