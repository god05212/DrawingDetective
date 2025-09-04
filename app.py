# 정답 마스킹 해야함

!pip -q install --upgrade openai gradio rapidfuzz pillow
import os, io, base64, re
from getpass import getpass
from openai import OpenAI
from PIL import Image
from rapidfuzz.distance import Levenshtein
import gradio as gr

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = getpass("OPENAI_API_KEY: ")
client = OpenAI()

def normalize(txt: str) -> str:
    if not txt: return ""
    t = txt.strip().lower()
    t = re.sub(r"[^\w가-힣 ]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t

def build_prompt(answer: str, hint: str) -> str:
    # 정답 단어가 힌트에 포함되면 [비밀]로 처리
    cleaned = hint
    for tok in set(normalize(answer).split()):
        if tok:
            cleaned = re.sub(tok, "[비밀]", cleaned, flags=re.IGNORECASE)
    guard = "Child-friendly, no text or letters in the image."
    return f"{guard}\nScene based on hint: {cleaned}. Bright, clean style."

def transcribe(path: str) -> str:
    with open(path, "rb") as f:
        # Whisper API를 사용하여 오디오 파일을 텍스트로 변환
        r = client.audio.transcriptions.create(model="whisper-1", file=f, response_format="text")
    return r

def gen_image(prompt: str) -> str:
    # 이미지 생성
    print(f"Prompt: {prompt}")  # 프롬프트 출력
    try:
        out = client.images.generate(model="dall-e-3", prompt=prompt, size="1024x1024")
        print(f"API Response: {out}")  # API 응답 확인

        # b64_json 대신 URL을 반환
        img_url = out.data[0].url
        return img_url  # URL 반환
    except Exception as e:
        print(f"Error: {str(e)}")  # 예외 발생 시 오류 출력
        return f"오류 발생: {str(e)}"

def judge(guess: str, answer: str, dist: int = 1):
    g, a = normalize(guess), normalize(answer)
    if not g: return "정답을 입력해 주세요."
    if g == a or Levenshtein.distance(g, a) <= dist:
        return "정답! 🎉"
    return "오답입니다. 힌트를 더 말해 보세요."

# Gradio 인터페이스 정의
with gr.Blocks(title="그림 탐정") as demo:
    gr.Markdown("### 그림 탐정\n마이크로 **힌트 말하기 → 전사 → 이미지 생성 → 정답 판정** 한 사이클")

    # 여기서 텍스트박스의 값을 자유롭게 입력할 수 있도록 수정합니다.
    answer = gr.Textbox(label="정답(호스트만 입력; 예: 고양이)", placeholder="정답을 입력하세요", type="text")

    with gr.Row():
        audio = gr.Audio(sources=["microphone"], type="filepath", label="🎤 힌트 녹음")
        stt_btn = gr.Button("1) 전사")

    hint = gr.Textbox(label="전사 결과(편집 가능)")

    gen_btn = gr.Button("2) 이미지 생성")
    img = gr.Image(label="AI 이미지")

    with gr.Row():
        guess = gr.Textbox(label="3) 정답 입력")
        judge_btn = gr.Button("채점")

    result = gr.Markdown()

    # 오디오 파일을 텍스트로 전사
    def on_stt(path):
        if not path: return ""
        return transcribe(path)

    # 이미지 생성
    def on_gen(ans, h):
        if not h: return None
        prompt = build_prompt(ans, h)
        img_url = gen_image(prompt)
        return img_url  # URL 반환

    # 정답 판정
    def on_judge(ans, g):
        return judge(g, ans)

    stt_btn.click(on_stt, audio, hint)
    gen_btn.click(on_gen, [answer, hint], img)
    judge_btn.click(on_judge, [answer, guess], result)

# 앱 실행
demo.launch(share=True)
