import streamlit as st
import os
import time
from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew, Process
from huggingface_hub import InferenceClient
from fpdf import FPDF

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# --- UI STYLING ---
st.set_page_config(page_title="Studio Architect | Agentic AI", page_icon="🏛️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    .report-title { font-size: 50px; font-weight: bold; color: #f0f2f6; }
    </style>
    """, unsafe_allow_html=True)

def generate_architectural_image(vision_text):
    # Use the token from the sidebar input if available, else from .env
    token = st.session_state.get("manual_hf_token") or HF_TOKEN
    if not token:
        return None
    
    client = InferenceClient(api_key=token)
    img_prompt = f"Hyper-realistic architectural photography, {vision_text}, luxury resort, 8k, daylight, professional hospitality marketing, cinematic lighting."
    
    image = client.text_to_image(img_prompt, model="black-forest-labs/FLUX.1-schnell")
    return image

# --- AGENT LOGIC ---
def run_agency(user_input):
    if not GEMINI_API_KEY:
        return "Error: GOOGLE_API_KEY not found."
    
    os.environ["OTEL_SDK_DISABLED"] = "true"

    # 1. Initialize LLM with optimized settings for stability
    gemini_llm = LLM(
        model="gemini/gemini-3-flash-preview", # Flash is more resilient to 503 errors than Pro
        api_key=GEMINI_API_KEY,
        temperature=0.7
    )

    # 2. Define Agents
    researcher = Agent(
        role='Heritage & Design Researcher',
        goal=f'Identify materials and styles for {user_input}',
        backstory='Expert in luxury architectural materials and regional heritage.',
        llm=gemini_llm,
        max_iter=3,
        max_rpm=2, # Conservative RPM to avoid 503s
        allow_delegation=False,
        verbose=True
    )

    architect = Agent(
        role='Principal Architect',
        goal='Draft a luxury architectural narrative and design brief',
        backstory='A world-renowned architect famous for blending nature with modern luxury.',
        llm=gemini_llm,
        max_rpm=2,
        allow_delegation=False,
        verbose=True
    )

    # 3. Define Tasks 
    research_task = Task(
        description=f"Identify 3 key authentic materials for: {user_input}.",
        expected_output="A list of 3 recommended materials with brief rationales.",
        agent=researcher
    )

    design_task = Task(
        description=f"Using the research, create a 3-paragraph architectural design brief for {user_input}.",
        expected_output="A professional 3-paragraph architectural concept narrative.",
        agent=architect
    )

    # 4. Form the Crew
    crew = Crew(
        agents=[researcher, architect],
        tasks=[research_task, design_task],
        process=Process.sequential,
        verbose=True,
        max_rpm=2 
    )

    # 5. EXECUTION WITH RETRY LOGIC (The 503 Fix)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return crew.kickoff()
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                st.warning(f"Gemini is busy. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

# --- MAIN UI ---
if "architectural_result" not in st.session_state:
    st.session_state.architectural_result = None
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
    """
    <h2 style='text-align: center; color: white; font-size: 28px; white-space: nowrap;'>
        Multi-Agent AI for Advanced Hospitality Design
    </h2>
    """, 
    unsafe_allow_html=True)

with col2:
    st.sidebar.title("🛠️ System Status")
    if GEMINI_API_KEY:
        st.sidebar.success("Gemini API: Active")
    if HF_TOKEN:
        st.sidebar.success("HuggingFace: Active")
    else:
        st.sidebar.error("HuggingFace Token Missing")

st.divider()
user_vision = st.text_input("Describe your vision", placeholder="e.g., A boutique heritage hotel in a restored Rajasthani palace...")

if st.button("Generate Architectural Concept"):
    if user_vision:
        # Reset previous results
        st.session_state.architectural_result = None
        st.session_state.generated_image = None 
        
        with st.status("🚀 Agents Collaborating...", expanded=True) as status:
            try:
                result = run_agency(user_vision)
                st.session_state.architectural_result = result
                status.update(label="✅ Concept Finalized!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="❌ Error Occurred", state="error")
                st.error(f"Execution failed: {str(e)}")
    else:
        st.warning("Please enter a vision to start.")

# --- RESULTS DISPLAY ---
if st.session_state.architectural_result:
    result = st.session_state.architectural_result
    
    st.subheader("Final Architectural Concept")
    st.markdown(result.raw)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.session_state.architectural_result:
            st.info("Design Concept Ready")            
    with c2:
        if st.button("Generate 3D Image"):
            with st.spinner("Painting visual concept"):
                try:
                    img = generate_architectural_image(user_vision)
                    st.session_state.generated_image = img
                except Exception as e:
                    st.error(f"Image Generation Error: {e}")

    if st.session_state.generated_image:
        st.divider()
        _, img_col, _ = st.columns([1, 2, 1]) 
        with img_col:
            st.image(
                st.session_state.generated_image, 
                caption="AI Generated 3D Render", 
                use_container_width=False, # This prevents it from stretching to full width
                width=500 # Manually setting a smaller width for better UI balance
            )