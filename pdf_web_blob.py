import pdfplumber
import streamlit as st
from azure.storage.blob import BlobServiceClient

# Azure Storage Account details
azure_storage_account_name = "storage09170285"
azure_storage_account_key = "5sOn/7pLTJBqdoSWi01788OBkRAtvJu3/MbEyuIQRKWWWSNT4PGIQkUlPmm/gWYRDu1egqdT5H4++AStHqL4UQ=="
container_name = "image"

# Function to upload file to Azure Storage
def upload_to_azure_storage(file):
    blob_service_client = BlobServiceClient.from_connection_string(f"DefaultEndpointsProtocol=https;AccountName={azure_storage_account_name};AccountKey={azure_storage_account_key}")
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=file.name)
    blob_client.upload_blob(file)

st.title("Azure Storage Uploader")

uploaded_file = st.file_uploader('Choose a PDF file', type="pdf")

if uploaded_file is not None:

    if st.button("Upload to Azure Storage"):
        try:
            upload_to_azure_storage(uploaded_file)
            st.success("File uploaded to Azure Storage!")
            pdf = pdfplumber.open(uploaded_file)

            content = ""
            for page in pdf.pages:
                text = page.extract_text()
                content += text
            
            st.write(content)
        except:
            st.error("Error uploading file to Azure Storage, because the file is already exist!")
