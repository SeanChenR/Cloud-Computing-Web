# Cloud Computing Web
<pre><code>https://52.146.52.174:8501/</code></pre>

## Environment
![Static Badge](https://img.shields.io/badge/1.8.2-FFDC35?style=plastic&label=Poetry&labelColor=black)
![Static Badge](https://img.shields.io/badge/3.11.7-0E76B6?style=plastic&label=Python&labelColor=black)
![Static Badge](https://img.shields.io/badge/0.2.1-007500?style=plastic&label=langchain&labelColor=black)
![Static Badge](https://img.shields.io/badge/0.2.1-FFBD9D?style=plastic&label=langchain-community&labelColor=black)
![Static Badge](https://img.shields.io/badge/4.2.0-5CADAD?style=plastic&label=pypdf&labelColor=black)
![Static Badge](https://img.shields.io/badge/1.9.1-B1BFE7?style=plastic&label=qdrant-client&labelColor=black)
![Static Badge](https://img.shields.io/badge/1.34.0-FF4B4B?style=plastic&label=streamlit&labelColor=black)
![Static Badge](https://img.shields.io/badge/0.3.12-00DB00?style=plastic&label=streamlit-option-menu&labelColor=black)
![Static Badge](https://img.shields.io/badge/0.4.2-8600FF?style=plastic&label=streamlit-extras&labelColor=black)
![Static Badge](https://img.shields.io/badge/1.0.0.b1-FF0080?style=plastic&label=azure-ai-translation-text&labelColor=black)
![Static Badge](https://img.shields.io/badge/0.9.0-FF8000?style=plastic&label=azure-cognitiveservices-vision-computervision&labelColor=black)
![Static Badge](https://img.shields.io/badge/12.20.0-9393FF?style=plastic&label=azure-storage-blob&labelColor=black)

## Tools
![Static Badge](https://img.shields.io/badge/Azure-0062AD?style=for-the-badge&logo=microsoftazure&labelColor=black)
![Static Badge](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&labelColor=black)
![Static Badge](https://img.shields.io/badge/Llama3-0467DF?style=for-the-badge&logo=meta&labelColor=black)
![Static Badge](https://img.shields.io/badge/Embedding-7B7B7B?style=for-the-badge&logo=vectorworks&labelColor=black)
![Static Badge](https://img.shields.io/badge/Qdrant-4F46DC?style=for-the-badge&logo=qase&labelColor=black)

## Introduction
This is a multi-functional intelligent application project that integrates various technologies, including **Image Recognize**, **Translation**, **PDF RAG**, **ChatBot**, and **Article Summarization**. We used tools such as translation, image recognition, and storage accounts in Azure, and also connected the llama3 model and Qdrant vector database, and used the Python streamlit to create a web that allows user interaction.

## Features
- **Image Recognition**: Automatically recognize and extract text and object information from images.
- **Translation**: Support for automatic translation in multiple languages, providing accurate translation results.
- **PDF RAG Technology**: Utilize Retrieval-Augmented Generation (RAG) technology for PDF document processing and information retrieval.
- **Chatbot**: An intelligent chatbot capable of natural language conversations and providing relevant information.
- **Article Summarization**: Automatically generate article summaries, extracting key content from articles.

## Azure Service
- **Azure Translation**: [Azure Translation Service](https://azure.microsoft.com/en-us/services/cognitive-services/translator/)
- **Azure Computer Vision**: [Azure Computer Vision Service](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/)
- **Azure Storage**: [Azure Storage Services](https://azure.microsoft.com/en-us/services/storage/)

## Model & Database
- TWCC LLM Model : https://tws.twcc.ai/service/ffm-llama3/
- TWCC Embedding Model : https://tws.twcc.ai/service/embedding/
- Qdrant cloud - Use API to connect with Qdrant cloud

## Enter Information About The Services & Tools You Use
Open the config.ini file and fill in the required information according to the corresponding title.

## Install The Required Dependencies & Run The Code
<pre><code>pip install -r requirements.txt</code></pre>
<pre><code>streamlit run "your file"</code></pre>
