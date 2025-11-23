import streamlit as st
from dotenv import load_dotenv
import os, faiss, numpy as np, asyncio
from sentence_transformers import SentenceTransformer
from openai import OpenAI, AsyncOpenAI
import streamlit as st
from PIL import Image
import base64, io
import speech_recognition as sr  
from gtts import gTTS               
import pygame                      
import tempfile
import streamlit.components.v1 as components

st.set_page_config(page_title="文渊博物馆智能导览", page_icon="🖼")
load_dotenv()
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
aclient = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")

# 读取并构建向量库
@st.cache_resource
def load_data():
    with open("museum_data.txt", "r", encoding="utf-8") as f:
        raw = f.read().strip().split("\n\n")
        docs = [b.strip() for b in raw if b.strip()]
    
    if not os.path.exists("faiss.index"):
        embeddings = embedder.encode(docs, normalize_embeddings=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, "faiss.index")
        np.save("docs.npy", np.array(docs, dtype=object))
    else:
        index = faiss.read_index("faiss.index")
        docs = np.load("docs.npy", allow_pickle=True).tolist()
    
    return docs, index

docs, index = load_data()

# 界面

st.title("🖼 文渊博物馆 · 智能导览员")
# ===== 拍照识文物（2025.11.23 万能版，永不报错）=====
st.markdown("### 📸 拍一张文物照片，我来告诉你它是谁")
uploaded_file = st.file_uploader("上传文物照片（支持任何格式）", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")   # ← 关键！强制转成 RGB
    st.image(image, caption="您上传的文物", width=300)
    
    with st.spinner("正在用通义千问多模态模型识别..."):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)  # 现在一定能存
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        response = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}},
                    {"type": "text", "text": f"这是文渊博物馆的哪件展品？请结合我提供的资料判断（只回答最匹配的一件）：\n" + "\n\n".join(docs)}
                ]
            }],
            temperature=0.3,
            max_tokens=500
        )
        result = response.choices[0].message.content
        st.success("识别结果：")
        st.markdown(result)
# ===== 第2天：语音输入 + 女声播报 =====
# ===== 云端语音输入（Web Speech API，Streamlit Cloud 完美支持）=====
st.markdown("### 🎤 语音问我（浏览器自动识别）")

if st.button("🎤 点我说话", key="web_voice"):
    st.write("请允许浏览器访问麦克风...")
    
    # JavaScript 代码（浏览器内置语音识别，无需 pyaudio）
    js_code = '''
    <script>
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.lang = 'zh-CN';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        
        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            parent.postMessage({type: 'streamlit:setComponentValue', value: transcript}, '*');
        };
        recognition.onerror = function(event) {
            parent.postMessage({type: 'streamlit:setComponentValue', value: '识别失败: ' + event.error}, '*');
        };
        recognition.start();
    } else {
        st.write("请用 Chrome 或 Edge 浏览器");
    }
    </script>
    '''
    st.components.v1.html(js_code, height=0)
    
    # 接收结果（用 session_state 监听）
    if 'voice_result' not in st.session_state:
        st.session_state.voice_result = ''
    
    voice_text = st.text_input("识别结果（自动填入）", value=st.session_state.voice_result, key="voice_output")
    
    if voice_text and voice_text != '识别失败: ':
        st.success(f"我听到你说：{voice_text}")
        
        # 直接触发多智能体回答（用 voice_text 替换 prompt）
        with st.chat_message("user"):
            st.markdown(voice_text)
        with st.chat_message("assistant"):
            with st.spinner("3位AI导游正在讨论..."):
                # 你的检索 + 多智能体代码（保持不变）
                query_vec = embedder.encode([voice_text], normalize_embeddings=True)
                D, I = index.search(query_vec, k=3)
                context = "\n\n".join([f"【资料{i+1}】\n{docs[i]}" for i, idx in enumerate(I[0])])
                
                expert = client.chat.completions.create(
                    model="qwen-max",
                    messages=[{"role": "user", "content": f"资料：{context}\n问题：{voice_text}\n请专业讲解："}],
                    temperature=0.3
                ).choices[0].message.content
                
                story = client.chat.completions.create(
                    model="qwen-max",
                    messages=[{"role": "system", "content": "你是一个会讲睡前故事的导游"},
                              {"role": "user", "content": f"讲成故事：{expert}"}],
                    temperature=0.7
                ).choices[0].message.content
                
                english = client.chat.completions.create(
                    model="qwen-max",
                    messages=[{"role": "user", "content": f"翻译成英文：{expert}"}],
                    temperature=0.3
                ).choices[0].message.content
                
                final_answer = f"**专业讲解：**\n{expert}\n\n**故事版：**\n{story}\n\n**English：**\n{english}"
                st.markdown(final_answer)
                
                # 女声播报（你已有的终极版）
                play_tts_final(final_answer)
        
        st.session_state.voice_result = ''  # 清空
st.caption("已加载展品数量："+str(len(docs))+" 件  │  模型：通义千问 Qwen-Max")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！我是文渊博物馆AI导览员，请问您想了解哪件展品？"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("问我任何关于展品的问题～"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("3位AI导游正在集体讨论..."):
            # 1. 先检索（你原来就有的代码，保留）
            query_vec = embedder.encode([prompt], normalize_embeddings=True)
            D, I = index.search(query_vec, k=3)
            context = "\n\n".join([f"【资料{i+1}】\n{docs[i]}" for i, idx in enumerate(I[0])])
            
            # 2. 多智能体开始！
            expert = client.chat.completions.create(
                model="qwen-max",
                messages=[{"role": "user", "content": f"资料：{context}\n问题：{prompt}\n请用专业语气详细讲解这件文物："}],
                temperature=0.3
            ).choices[0].message.content
            
            story = client.chat.completions.create(
                model="qwen-max",
                messages=[{"role": "system", "content": "你是一个会讲睡前故事的导游，要生动有趣"},
                          {"role": "user", "content": f"把这件文物讲成一个吸引人的睡前故事：{expert}"}],
                temperature=0.7
            ).choices[0].message.content
            
            english = client.chat.completions.create(
                model="qwen-max",
                messages=[{"role": "user", "content": f"把这段翻译成自然流利的英文：{expert}"}],
                temperature=0.3
            ).choices[0].message.content
            
            final_answer = f"**【专业讲解】**\n{expert}\n\n**【睡前故事版】**\n{story}\n\n**【English Guide】**\n{english}"
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            import edge_tts
            import pygame
            import tempfile
            import time
            import os

            def play_tts_final(text):
                # 先生成文件
                communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                communicate.save_sync(tmp_file.name)
                tmp_path = tmp_file.name
                tmp_file.close()  # 先关闭句柄

                # 延迟 0.3 秒确保文件完全写入
                time.sleep(0.3)

                # 播放
                pygame.mixer.init()
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                
                # 阻塞等待播完
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                # 播完再删文件
                try:
                    os.unlink(tmp_path)
                except:
                    pass  # 删不掉也没事

            st.write("正在用微软晓晓女声播报...")
            play_tts_final(final_answer)
