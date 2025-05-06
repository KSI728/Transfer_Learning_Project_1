import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
from gtts import gTTS
import tempfile

# 페이지 설정 및 스타일
st.set_page_config(
    page_title="어린이 분리수거 도우미 AI", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS 스타일 적용
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.2rem;
        text-align: center;
        color: #4CAF50;
        margin-bottom: 2rem;
    }
    .result-container {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .result-title {
        font-size: 1.5rem;
        color: #2E7D32;
        margin-bottom: 10px;
    }
    .instruction-container {
        background-color: #FFF8E1;
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
    }
    .emoji-icon {
        font-size: 2rem;
        margin-right: 10px;
    }
    .upload-section {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        color: #757575;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# 클래스 및 설명 정의 (한글 클래스명과 이모지 추가)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
korean_names = {
    'cardboard': '종이상자',
    'glass': '유리병',
    'metal': '캔류',
    'paper': '종이',
    'plastic': '플라스틱류',
    'trash': '일반쓰레기'
}
class_emojis = {
    'cardboard': '📦',
    'glass': '🥛',
    'metal': '🥫',
    'paper': '📄',
    'plastic': '🧴',
    'trash': '🗑️'
}
explanations = {
    "cardboard": "종이상자는 테이프를 떼고 접어서 종이로 분리해 버려요!",
    "glass": "유리병은 병뚜껑을 제거하고 투명/갈색/기타 유리로 나누어 버려요.",
    "metal": "캔과 고철은 물로 헹군 후 압착해서 버리면 좋아요.",
    "paper": "종이는 물에 젖지 않게 모아서 종이류로 버려요.",
    "plastic": "플라스틱은 라벨을 떼고 깨끗이 씻어 플라스틱으로 분리해요.",
    "trash": "이건 일반쓰레기야! 재활용이 안 되는 쓰레기들은 종량제 봉투에 넣자."
}
recycling_tips = {
    "cardboard": "물에 젖은 종이상자는 재활용이 어려워요.",
    "glass": "깨진 유리는 신문지에 싸서 일반쓰레기로 버려야 해요.",
    "metal": "알루미늄 호일은 깨끗이 씻어서 재활용해요.",
    "paper": "스프링이 있는 노트는 스프링을 제거해야 해요.",
    "plastic": "페트병 라벨은 반드시 떼서 분리배출해요.",
    "trash": "음식물이 묻은 쓰레기는 깨끗이 씻어서 버려요."
}

# 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 모델 로드 함수 (캐싱)
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(2048, 6)  # 출력 클래스 수 맞추기
    model.load_state_dict(torch.load("best_model.pt", map_location='cpu'))
    model.eval()
    return model

# 예측 함수
def predict(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

# 헤더 표시
st.markdown("<div class='main-title'>♻️ 어린이 분리수거 도우미 AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>쓰레기 사진을 올리면 AI가 분리수거 방법을 알려줘요!</div>", unsafe_allow_html=True)

# 3단계 간단 설명
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 1️⃣ 사진 찍기")
    st.markdown("쓰레기 사진을 찍어요")
with col2:
    st.markdown("### 2️⃣ 업로드")
    st.markdown("찍은 사진을 올려요")
with col3:
    st.markdown("### 3️⃣ 결과 확인")
    st.markdown("분리수거 방법을 배워요")


# 모델 로드
try:
    model = load_model()
except Exception as e:
    st.error(f"모델을 불러오는 중 오류가 발생했어요: {e}")

# 파일 업로드만 사용 (모바일·데스크톱 공용)
uploaded_file = st.file_uploader(
    "📸 갤러리나 파일에서 사진을 선택해주세요",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

# 예측 및 결과 표시
if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        
        # 이미지 표시
        st.image(image, caption="업로드한 이미지", use_container_width=True)
        
        # 예측 실행
        with st.spinner('AI가 열심히 생각하고 있어요...'):
            predicted_class = predict(image)
            korean_name = korean_names[predicted_class]
            emoji = class_emojis[predicted_class]
            explanation = explanations[predicted_class]
            tip = recycling_tips[predicted_class]
        
        # 결과 표시
        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-title'>{emoji} 이것은 <span style='color:#FF5722;'>{korean_name}</span>이에요!</div>", unsafe_allow_html=True)
        
        # 분리수거 방법 안내
        st.markdown("<div class='instruction-container'>", unsafe_allow_html=True)
        st.markdown(f"#### 📝 분리수거 방법")
        st.markdown(f"{explanation}")
        st.markdown(f"#### 💡 알면 좋은 팁!")
        st.markdown(f"{tip}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 음성 재생
        from io import BytesIO

        import base64
        
        # 1) TTS 생성
        tts = gTTS(text=f"이것은 {korean_name}이에요! {explanation}", lang='ko')

        # 2) 메모리 버퍼에 MP3 기록
        audio_buf = BytesIO()
        tts.write_to_fp(audio_buf)
        audio_buf.seek(0)

        # 3) Base64로 인코딩
        b64_str = base64.b64encode(audio_buf.read()).decode("utf-8")

        # 4) HTML 오디오 태그를 인라인 데이터로 삽입
        audio_html = f"""
        #### 🔊 AI 선생님의 말씀을 들어보세요!
        <audio controls="controls">
        <source src="data:audio/mp3;base64,{b64_str}" type="audio/mp3" />
        Your browser does not support the audio element.
        </audio>
        """

        st.markdown(audio_html, unsafe_allow_html=True)
        
        # 학습용 분리수거 이미지 추가
        st.markdown("#### 올바른 분리수거 방법 알아보기")
        if predicted_class == 'cardboard':
            st.image(r"C:\Users\a\Desktop\KDT7\TRANSFER_LEARNING\project\data\종이박스.png", caption="종이상자 분리수거 방법", use_container_width=True)
        elif predicted_class == 'glass':
            st.image(r"C:\Users\a\Desktop\KDT7\TRANSFER_LEARNING\project\data\유리병류.jpg", caption="유리병류 분리수거 방법", use_container_width=True)
        elif predicted_class == 'metal':
            st.image(r"C:\Users\a\Desktop\KDT7\TRANSFER_LEARNING\project\data\캔류.jpg", caption="캔류 분리수거 방법", use_container_width=True)
        elif predicted_class == 'paper':
            st.image(r"C:\Users\a\Desktop\KDT7\TRANSFER_LEARNING\project\data\종이류.jpg", caption="종이 분리수거 방법", use_container_width=True)
        elif predicted_class == 'plastic':
            st.image(r"C:\Users\a\Desktop\KDT7\TRANSFER_LEARNING\project\data\플라스틱류.jpg", caption="플라스틱류 분리수거 방법", use_container_width=True)
        elif predicted_class == 'trash':
            st.image(r"C:\Users\a\Desktop\KDT7\TRANSFER_LEARNING\project\data\일반쓰레기.jpg", caption="일반쓰레기 버리는 방법", use_container_width=True)

    except Exception as e:
        st.error(f"이미지 처리 중 오류 발생: {e}")
else:
    # 처음 화면에 하나의 이미지만 보여주기
    st.markdown("#### 분리수거를 배워봐요! 📦 🥛 🥫 📄 🧴 🗑️")
    st.image(r"C:/Users/a/Desktop/KDT7/TRANSFER_LEARNING/project/data/분리수거배워봐요.png", 
             caption="분리수거 배우기", 
             use_container_width=True)
    
    st.info("분리수거할 쓰레기 이미지를 업로드해주세요 😊")

# 푸터
st.markdown("<div class='footer'>어린이 분리수거 도우미 AI - 지구를 지키는 작은 실천 🌏</div>", unsafe_allow_html=True)