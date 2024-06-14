import time
import requests
import configparser
import streamlit as st
from bs4 import BeautifulSoup
from datetime import datetime as dt
from qdrant_client import QdrantClient
from streamlit_extras.grid import grid
from streamlit_option_menu import option_menu
from langchain_qdrant import Qdrant
from langchain.chains import ConversationChain
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from blob_list import get_blob_list
from vision import recognizer_azure
from translator import translator_azure
from storage_upload import upload_to_azure_storage
from TWSC_embedding import get_embeddings_model
from llm import chat_ffm, memory

config = configparser.ConfigParser()
config.read('config.ini')

API_KEY = config['embedding']['API_KEY']
API_URL = config['embedding']['API_URL']
MODEL_NAME = config['embedding']['MODEL_NAME']

embeddings_model = get_embeddings_model(API_URL, API_KEY, MODEL_NAME)

qdrant_url = config['qdrant']['url']
qdrant_port = config['qdrant']['port']
qdrant_api_key = config['qdrant']['api_key']

background_color = "green"

language = {
    "繁體中文 (zh-hant)" : "zh-hant",
    "英文 (en)" : "en",
    "日文 (ja)" : "ja",
    "韓文 (ko)" : "ko",
    "西班牙文 (es)" : "es",
}

def rag(query, rag_result):
    prompt_template = """你是一位樂於助人的小幫手，請皆以繁體中文回答問題，並僅根據<text></text>這個html tag中的參考資料回答問題，不知道就說不知道，不準依照自己的想法回答。
                            然後請以一個專業人士或相關單位工作人員的角度回答問題，若資料有出處請註明出處。
                            <text>{context}</text>
                            問題：{question}"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    llm = prompt | chat_ffm
    result = llm.invoke({"context":rag_result, "question":query})
    return result.content

def single_chat(query):
    prompt_template = """你是一位得力AI助手，對任何問題總能回應有幫助的答案。
                            使用者輸入內容：{question}"""
    prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
    llm = prompt | chat_ffm
    result = llm.invoke({"question":query})
    return result.content

def memory_chat(query):
    conversation_with_summary = ConversationChain(
        llm=chat_ffm,
        memory=memory,
        verbose=True,
    )
    
    ans = conversation_with_summary.predict(input=query)
    return ans

def summary(query):
    prompt_template = "I need to summarize the article. No prefix! The article is as follows: {article}"
    prompt = PromptTemplate(input_variables=["article"], template=prompt_template)
    llm = prompt | chat_ffm
    result = llm.invoke({"article":query})
    return result.content

def typewriter(text: str, speed: int):
    tokens = text.split()
    container = st.empty()
    for index in range(len(tokens) + 1):
        curr_full_text = " ".join(tokens[:index])
        container.markdown(curr_full_text)
        time.sleep(1 / speed)

def txt_contents(url):
    response = requests.get(url)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.prettify()

with st.sidebar:
    selected = option_menu(
    menu_title="Main Menu",
    options=["Home", "PDF QA", "Translate", "Summarize", "Chat Bot", "Image Vision"],
    icons=["house-fill", "filetype-pdf", "translate", "feather", "robot", "cpu"],
    menu_icon="list-ul",
    default_index=0
)

client = QdrantClient(url=qdrant_url,
                        port=qdrant_port,
                        api_key=qdrant_api_key)

if selected == "Home":
    st.header(':rainbow[雲端運算服務_第六組_期末專案]')

    st.subheader("👈 可以透過左側列表選擇服務")

    st.markdown("### Home")

    st.code(
    """
    🔧 工具與選單服務介紹
    🔗 串接台智雲、Qdrant Cloud
    ☁️ Azure - Translate Computer Vision Storage 
    """
    )

    st.markdown("### PDF QA")

    st.code(
    """
    🔝 上傳PDF檔至資料庫
    📑 選擇資料庫的PDF檔，並顯示內容
    💡 根據PDF檔的內容進行提問，AI會根據PDF檔回答問題
    """
    )

    st.markdown("### Translate")

    st.code(
    """
    🇹🇼🇯🇵🇰🇷🇺🇸🇪🇸 五種國際語言可選擇
    """
    )

    st.markdown("### Summarize")

    st.code(
    """
    📍 上傳TXT檔至資料庫
    📰 輸入文章或新聞，AI會對其進行摘要
    💻 選擇資料庫的TXT檔，並讓AI進行摘要
    """
    )

    st.markdown("### Chat Bot")

    st.code(
    """
    🔑 像ChatGPT使用即可
    """
    )

    st.markdown("### Image Vision")

    st.code(
    """
    🐶 上傳jpg/png檔至資料庫，並對其進行辨識
    🐮 選擇資料庫的圖片檔，並讓AI對其進行辨識
    """
    )

if selected == "PDF QA":
    st.header(':rainbow[雲端運算服務_第六組_PDF QA]')

    selected_pdf = option_menu(
        menu_title=None,
        options=["PDF File", "PDF View", "Upload"],
        icons=["file-pdf-fill", "printer-fill", "cloud-upload-fill"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": background_color},
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {
                "font-size": "25px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#6F00D2"},
        },
    )

    if selected_pdf == "PDF View":
        blob_list = get_blob_list("pdf")
        pdf_file = st.selectbox("請選擇pdf檔", blob_list)
        url = f"https://storage09170285.blob.core.windows.net/image/{pdf_file}"

        if st.button("Print the contents of PDF", use_container_width=True):
            loader = PyPDFLoader(url)
            docs = loader.load()
            for doc in docs:
                st.write(doc.page_content)

    if selected_pdf == "PDF File":
        blob_list = get_blob_list("pdf")
        pdf_grid = grid(1,1,1, vertical_align="center")
        pdf_file = pdf_grid.selectbox("請選擇pdf檔", blob_list)
        query = pdf_grid.text_input("請輸入關於此pdf檔的相關問題")
        if st.button("Submit the question", use_container_width=True):
            with st.spinner('Searching from Qdrant Cloud...'):
                search_result = client.search(
                    collection_name=pdf_file,
                    query_vector=embeddings_model.embed_query(query),
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )
                if len(search_result) == 0:
                    with st.chat_message("assistant"):
                        st.markdown("No Results Found")
                else:
                    llm_result = rag(query, search_result[0].payload['page_content'])
                    with st.chat_message("assistant"):
                        st.markdown(llm_result)

    if selected_pdf == "Upload":
        uploaded_file = st.file_uploader('Choose a PDF file', type="pdf")

        if uploaded_file is not None:
            collection_name = uploaded_file.name
            url = f"https://storage09170285.blob.core.windows.net/image/{uploaded_file.name}"
            if st.button("Upload to Azure Storage", use_container_width=True):
                try:
                    with st.spinner('Uploading to Azure...'):
                        upload_to_azure_storage(uploaded_file)
                        st.success("File uploaded to Azure Storage!", icon="✅")
                except:
                    st.error("Error uploading file to Azure Storage, because the file is already exist!")
                
                finally:
                    with st.spinner('Uploading to Qdrant Cloud...'):
                        loader = PyPDFLoader(url)

                        docs = loader.load()

                        splitter = RecursiveCharacterTextSplitter(    
                            chunk_size=500,
                            chunk_overlap=100)

                        chunks = splitter.split_documents(docs)

                        qdrant = Qdrant.from_documents(
                            chunks,
                            embeddings_model,
                            url=qdrant_url,
                            port=qdrant_port,
                            api_key=qdrant_api_key,
                            collection_name=collection_name,
                            force_recreate=True,
                        )
                        st.success("It is successfully imported them into the Qdrant database.", icon="✅")

if selected == "Translate":
    st.header(':rainbow[雲端運算服務_第六組_Translate]')
    my_grid = grid(2, 2, 1, vertical_align="bottom")
    # row1
    Input = my_grid.selectbox("選擇輸入語言：", ["繁體中文 (zh-hant)", "英文 (en)", "日文 (ja)", "韓文 (ko)", "西班牙文 (es)"])
    Output = my_grid.selectbox("選擇輸出語言：", ["英文 (en)", "繁體中文 (zh-hant)", "日文 (ja)", "韓文 (ko)", "西班牙文 (es)"])
    # row2
    text_area_guide = f"請用{Input}輸入欲翻譯的文字："
    if language[Input] == "zh-hant":
        text_area_guide_en = "你好！"
    elif language[Input] == "en":
        text_area_guide_en = "Hello !"
    elif language[Input] == "ja":
        text_area_guide_en = "こんにちは！"
    elif language[Input] == "ko":
        text_area_guide_en = "안녕하세요！"
    elif language[Input] == "es":
        text_area_guide_en = "Hola !"
    text_area_input = my_grid.text_area(text_area_guide, text_area_guide_en, height=40)
    if "default" not in st.session_state:
        st.session_state["default"] = ""
    my_area = my_grid.text_area("翻譯結果：", value=st.session_state["default"], height=40)
    #row3
    if my_grid.button("Translate", use_container_width=True, help="Press the Bottom to Translate"):
        with st.spinner('Translating...'):
            translate_result = translator_azure(language[Input], language[Output], text_area_input)
            st.session_state["default"] = translate_result
            st.rerun()

if selected == "Summarize":
    st.header(':rainbow[雲端運算服務_第六組_Summarize]')

    selected_summary = option_menu(
        menu_title=None,
        options=["TXT File", "Text Input", "Upload"],
        icons=["eye-fill", "chat-text", "cloud-upload-fill"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": background_color},
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {
                "font-size": "25px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#6F00D2"},
        },
    )

    if selected_summary == "TXT File":
        blob_list = get_blob_list("txt")
        txt_grid = grid(1,[1,1],1, vertical_align="center")
        txt_file = txt_grid.selectbox("請選擇文字檔", blob_list)
        url = f"https://storage09170285.blob.core.windows.net/image/{txt_file}"
        contents = txt_contents(url)
        txt_grid.text_area("The contents of the txt file：", value=contents, height=300)
        if "default_file" not in st.session_state:
            st.session_state["default_file"] = ""
        txt_grid.text_area("Summary Result：", value=st.session_state["default_file"], height=300)

        if txt_grid.button("Go Summary", use_container_width=True):
            with st.spinner('Reading and Summarzing...'):
                summary_result = summary(contents)
                st.session_state["default_file"] = summary_result
                st.rerun()

    if selected_summary == "Upload":
        uploaded_file = st.file_uploader('Choose a TXT file', type="txt")

        if uploaded_file is not None:
            url = f"https://storage09170285.blob.core.windows.net/image/{uploaded_file.name}"

            summary_grid = grid([1,1], 1, 1, vertical_align="bottom")
            if "default_txt" not in st.session_state:
                st.session_state["default_txt"] = ""
            summary_grid.text_area("The contents of the uploaded file：", value=st.session_state["default_txt"], height=300)
            if "default_summary" not in st.session_state:
                st.session_state["default_summary"] = ""
            summary_grid.text_area("Summary Result：", value=st.session_state["default_summary"], height=300)

            if st.button("Upload & Summarize", use_container_width=True):
                try:
                    with st.spinner('Uploading...'):
                        upload_to_azure_storage(uploaded_file)
                        st.success("File uploaded to Azure Storage!", icon="✅")
                except:
                    st.error("Error uploading file to Azure Storage, because the file is already exist!")

                finally:
                    with st.spinner('Summarzing...'):
                        time.sleep(5)
                        contents = txt_contents(url)
                        st.session_state["default_txt"] = contents
                        summary_result = summary(contents)
                        st.session_state['default_summary'] = summary_result
                        st.spinner('Reading and Summarzing...')
                        st.rerun()

    article = """
輝達（NVIDIA）執行長黃仁勳於6月2日在台大體育館開講，打響2024年台北國際電腦展（COMPUTEX）的第一槍。該場演說萬眾矚目，包含廣達董事長林百里、聯發科執行長蔡力行和美超微（Supermicro）董事長梁見後等科技大老皆到場出席。「3兆美元產值的IT產業，未來將會催出高達100兆美元的產業革命。」黃仁勳再度化身最強AI推銷員，除宣佈AI模型推論服務產品「NVIDIA NIM」，同時提出全新概念數位人類（Digital Human），大秀包含3D模擬和光線追蹤等技術底蘊。《數位時代》盤點黃仁勳演說中，不能錯過的AI三大關鍵字。

關鍵字一：NIM。推論微服務，讓你幾分鐘內就可用AI
黃仁勳重申了AI的發展歷程：2012年，一名多倫多大學的學生Alex Krizhevsky，運用兩張輝達顯卡和120萬張圖片進行AI建模，達到錯誤率僅15%的成績，和前一年的25%相比為飛躍式的進步。輝達也開始積極佈局AI，並於2006年開發出CUDA。這是一套輝達提供給開發人員的編程工具，工程師除了能省下大量撰寫低階語法的時間，還能直接使用高階語法諸如C++或Java等來編寫應用於通用GPU上的演算法，解決平行運算中複雜的問題。簡單來說，就是提供AI開發者更簡單的工具。

NIM為預訓練模型包，內含CUDA軟體，以及文字、語音或畫面等模型，使用方式就如ChatGPT般輸入指令即可。這讓開發者能在幾分鐘內建構如Copilot或聊天機器人等應用程式，或生成一段語句、影片或圖片。甚至於，透過NIM還能用於生物科技，加快藥物探索的進度。

關鍵字二：數位人類。AI更像真人了？
黃仁勳還提出「數位人類（Digital Human）」的概念，表示透過「NVIDIA ACE NIM」的服務，於教育、行銷或是遠距醫療等場域，輕鬆建立專屬的數位人類。現場展示下，數位人類開口說的神情和語氣，栩栩如生。

關鍵字三：Rubin。黃仁勳首次揭露下一代GPU路線
在硬體方面，輝達展示出攜手華碩及微星所推出的RTX AI PC，表示將有200多款產品問世。業內人士透露，輝達也將攜手聯發科，採台積電3奈米製程製作Arm架構的PC晶片，預期年底或明年上半量產。值得一提的是，黃仁勳今日首度揭露下一代GPU平台「Rubin」，預期將採用HBM4，並於2026年正式推出，2027年則將進入「Rubin Ultra」世代，預期將採用3奈米製程。2025年則將推出「Blackwell Ultra」，預期將採HBM3e。
"""
    if selected_summary == "Text Input":
        summary_grid = grid(1, 1, 1, vertical_align="bottom")
        summary_input = summary_grid.text_area("可以輸入新聞或文章！（字數請控制在500字內）", article.strip(), height=230)
        if summary_grid.button("Go Summary", use_container_width=True):
            with st.spinner('Reading and Summarzing...'):
                summary_result = summary(summary_input)
                tokens = summary_result.split()
                container = summary_grid.empty()
                for index in range(len(tokens) + 1):
                    curr_full_text = " ".join(tokens[:index])
                    container.markdown(curr_full_text)
                    time.sleep(1 / 10)

if selected == "Chat Bot":
    st.header(':rainbow[雲端運算服務_第六組_ChatBot]')

    selected_bot = option_menu(
        menu_title=None,
        options=["Memory", "Single Round"],
        icons=["memory", "chat-dots-fill"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": background_color},
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {
                "font-size": "25px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#6F00D2"},
        },
    )

    if selected_bot == "Memory":
        with st.chat_message("assistant"):
            st.markdown(":orange[您好，我可以在我能力範圍內盡可能幫助您！😎]")
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Chat With Llama3 Bot"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            response = memory_chat(prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

    if selected_bot == "Single Round":
        with st.chat_message("assistant"):
            st.markdown(":orange[您好，我可以在我能力範圍內盡可能幫助您！😎]")
        # Initialize chat history
        if "messages_single" not in st.session_state:
            st.session_state.messages_single = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages_single:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Chat With Llama3 Bot"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages_single.append({"role": "user", "content": prompt})

            response = single_chat(prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages_single.append({"role": "assistant", "content": response})

if selected == "Image Vision":
    st.header(':rainbow[雲端運算服務_第六組_Image Vision]')

    selected_image = option_menu(
        menu_title=None,
        options=["Image DataBase", "Upload"],
        icons=["file-richtext-fill", "cloud-upload-fill"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": background_color},
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {
                "font-size": "25px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#6F00D2"},
        },
    )

    if selected_image == "Image DataBase":
        blob_list = get_blob_list("image")
        vision_grid = grid([1,1],[1,1],1, vertical_align="center")
        image_file = vision_grid.selectbox("請選擇圖片", blob_list)
        vision_input_lan = vision_grid.selectbox("選擇輸出語言：", ["繁體中文 (zh-hant)", "英文 (en)", "日文 (ja)", "西班牙文 (es)"])
        url = f"https://storage09170285.blob.core.windows.net/image/{image_file}"
        vision_grid.image(url, width=300)
        if "default_vision_data" not in st.session_state:
            st.session_state["default_vision_data"] = ""
        vision_result = vision_grid.text_area("辨識結果：", value=st.session_state["default_vision_data"], height=50)

        if vision_grid.button("Recognize", use_container_width=True, help="Press the Bottom to Recognize"):
            with st.spinner('Recognizing...'):
                recognize_result, recognize_score = recognizer_azure(url, language[vision_input_lan], 1)
                st.session_state["default_vision_data"] = recognize_result
                st.rerun()

    if selected_image == "Upload":
        vision_grid = grid(1,1,[1,1], vertical_align="center")
        vision_input_lan = vision_grid.selectbox("選擇輸出語言：", ["繁體中文 (zh-hant)", "英文 (en)", "日文 (ja)", "西班牙文 (es)"])

        uploaded_file = vision_grid.file_uploader("Choose a image file!")
        if uploaded_file is not None:
            vision_grid.image(uploaded_file, width=300)
            if "default_vision" not in st.session_state:
                st.session_state["default_vision"] = ""
            vision_result = vision_grid.text_area("辨識結果：", value=st.session_state["default_vision"], height=50)
            url = f"https://storage09170285.blob.core.windows.net/image/{uploaded_file.name}"

            # Upload the file to Azure Storage on button click
            if st.button("Upload to Azure Storage and Recognize", use_container_width=True):
                try:
                    with st.spinner('Uploading...'):
                        upload_to_azure_storage(uploaded_file)
                        st.success("File uploaded to Azure Storage!", icon="✅")
                except:
                    st.error("Error uploading file to Azure Storage, because the file is already exist!")
            
                finally:
                    with st.spinner('Recognizing...'):
                        time.sleep(5)
                        recognize_result, recognize_score = recognizer_azure(url, language[vision_input_lan], 1)
                        st.session_state["default_vision"] = recognize_result
                        st.session_state["default_score"] = round(recognize_score*100, 2)
                        st.rerun()
