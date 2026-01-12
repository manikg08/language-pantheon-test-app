"""
Complete German B2 Vocabulary Learning Pipeline
Audio Transcription → Vocabulary Extraction → Interactive Learning
"""

import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from pathlib import Path

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================================
# STEP 1: AUDIO TRANSCRIPTION
# ============================================================================


def transcribe_audio_file(audio_file_path):
    """
    You are a German language transcription assistant, the audio provided to you is from a german language class conversation, where there is a mixed language of german, english and hindi being spoken. You should only transcribe the german parts of the audio, and ignore any english or hindi parts. Please provide the transcription in german language only, do not translate to english.

    Args:
        audio_file_path: Path to audio file

    Returns:
        Transcribed German text
    """
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="gpt-4o-transcribe",
                language="de",
                response_format="text",
                prompt="This is a conversation in German language.",
            )
        return transcription
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None


def transcribe_uploaded_audio(uploaded_file):
    """
    Transcribe audio from uploaded file.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Transcribed text
    """
    # Save uploaded file temporarily
    temp_path = f"temp_audio_{uploaded_file.name}"

    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Transcribe
        transcription = transcribe_audio_file(temp_path)

        # Clean up
        os.remove(temp_path)

        return transcription

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None


# ============================================================================
# STEP 2: VOCABULARY EXTRACTION
# ============================================================================


def extract_b2_vocabulary(german_text):
    """
    Extract important B2-level vocabulary from German text using OpenAI.

    Args:
        german_text: German text to analyze

    Returns:
        List of vocabulary dictionaries
    """
    prompt = f"""You are a German language teacher specializing in B1-level and above vocabulary.

Analyze the following German text and extract the most important words for B1-level and above learners.

Text:
{german_text}

For each important word, provide:
1. The word in its base form (infinitive for verbs, nominative singular for nouns)
2. English translation
3. Part of speech (noun, verb, adjective, adverb, etc.)
4. Article (for nouns: der/die/das, or empty string if not a noun)
5. An example sentence using the word in context
6. Why it's important for learners

Focus on:
- Advanced vocabulary (B1 level and above)
- Verbs with separable/inseparable prefixes
- Compound nouns
- Professional/academic vocabulary
- Skip basic A1/A2 words like "haben", "sein", "die", "der", "und", etc.

Return the response as a JSON array with this structure:
[
  {{
    "word": "beantworten",
    "translation": "to answer, to reply",
    "pos": "verb",
    "article": "",
    "example": "Ich beantworte die E-Mails jeden Morgen.",
    "b2_relevance": "Important separable prefix verb used in professional contexts"
  }},
  {{
    "word": "Kindergarten",
    "translation": "kindergarten",
    "pos": "noun",
    "article": "der",
    "example": "Die Kinder gehen jeden Tag in den Kindergarten.",
    "b2_relevance": "Common compound noun in family and education contexts"
  }}
]

Return ONLY the JSON array, no markdown formatting, no other text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a German language teaching assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        # Extract and parse JSON
        content = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        vocabulary = json.loads(content.strip())
        return vocabulary

    except Exception as e:
        st.error(f"Error extracting vocabulary: {e}")
        return []


# ============================================================================
# STEP 3: QUIZ GENERATION
# ============================================================================


def generate_quiz_question(word_data):
    """Generate a multiple choice quiz question for a vocabulary word."""
    prompt = f"""Create a multiple choice quiz question to test knowledge of the German word "{word_data['word']}" 
    (meaning: {word_data['translation']}).

Create:
1. A sentence in German with the word missing (use ___ for the blank)
2. 4 answer options (one correct, three plausible distractors that are similar words)
3. The index of the correct answer (0-3)
4. A brief explanation

Return as JSON:
{{
  "question": "Ich muss noch die E-Mails ___.",
  "options": ["beantworten", "antworten", "fragen", "sprechen"],
  "correct": 0,
  "explanation": "beantworten is correct because it means to answer/reply to something formally, commonly used with emails."
}}

Return ONLY the JSON, no other text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a German language quiz generator.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        content = response.choices[0].message.content.strip()

        # Clean JSON
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        return json.loads(content.strip())

    except Exception as e:
        st.error(f"Error generating quiz: {e}")
        return None


# ============================================================================
# STREAMLIT APP
# ============================================================================


def main():
    """Main Streamlit app with complete pipeline."""

    # Page configuration
    st.set_page_config(
        page_title="German B2 Learning Pipeline",
        page_icon="🇩🇪",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None
    )

    # Custom CSS - Professional Enterprise UI
    st.markdown(
        """
        <style>
        /* Global Styles - Professional Typography */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #1e293b;
            background: #f8fafc !important;
        }

        /* Override Streamlit's default backgrounds */
        .stApp {
            background: #f8fafc !important;
        }

        section[data-testid="stSidebar"] > div {
            background: #f8fafc !important;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            color: #0f172a !important;
            font-weight: 600;
        }

        /* Text elements - more selective */
        p {
            color: #1e293b;
        }

        label {
            color: #0f172a;
        }

        /* Form labels */
        .stRadio > label,
        .stSelectbox > label,
        .stTextArea > label,
        .stFileUploader > label,
        [data-testid="stWidgetLabel"] {
            color: #0f172a !important;
            background: transparent !important;
        }

        /* Main Container */
        .main {
            background: #f8fafc;
        }

        /* Force consistent backgrounds */
        .main .block-container {
            background: #f8fafc;
        }

        /* Ensure all content areas have proper background */
        [data-testid="stVerticalBlock"] {
            background: transparent;
        }

        [data-testid="stHorizontalBlock"] {
            background: transparent;
        }

        /* Header Styling - Professional */
        .main-header {
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
            padding: 2.5rem 3rem;
            margin-bottom: 2.5rem;
            border-bottom: 4px solid #1e40af;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        .main-title {
            color: white;
            font-size: 2.5rem;
            font-weight: 600;
            margin: 0;
            letter-spacing: -0.5px;
        }

        .main-subtitle {
            color: #e0e7ff;
            font-size: 1.1rem;
            margin-top: 0.5rem;
            font-weight: 400;
            letter-spacing: 0.3px;
        }

        /* Vocabulary Cards - Professional */
        .vocab-card {
            background: white;
            padding: 24px 28px;
            margin: 16px 0;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #2563eb;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
            transition: all 0.2s ease;
        }

        .vocab-card:hover {
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
            border-left-color: #1e40af;
        }

        .word-title {
            font-size: 26px;
            font-weight: 600;
            color: #0f172a;
            margin-bottom: 8px;
            letter-spacing: -0.3px;
        }

        .translation {
            font-size: 18px;
            color: #64748b;
            margin-bottom: 16px;
            font-weight: 400;
        }

        .pos-badge {
            display: inline-block;
            padding: 4px 12px;
            background: #dbeafe;
            color: #1e40af;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            border: 1px solid #bfdbfe;
        }

        /* Example Box - Professional */
        .example {
            background: #f1f5f9;
            padding: 16px 20px;
            border-left: 3px solid #2563eb;
            margin: 12px 0;
            font-size: 15px;
            line-height: 1.7;
            border: 1px solid #e2e8f0;
        }

        .example strong {
            color: #1e40af;
            font-weight: 600;
        }

        /* Transcript Box - Professional */
        .transcript-box {
            background: white;
            padding: 24px 28px;
            border: 1px solid #e2e8f0;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
            line-height: 1.8;
            font-size: 15px;
            color: #334155;
        }

        /* Metric Cards - Professional Dashboard Style */
        .metric-card {
            background: white;
            padding: 24px;
            text-align: center;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
            transition: all 0.2s ease;
        }

        .metric-card:hover {
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
            border-color: #2563eb;
        }

        .metric-card h3 {
            color: #2563eb;
            font-size: 2.25rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -1px;
        }

        .metric-card p {
            color: #64748b;
            font-size: 0.875rem;
            margin: 8px 0 0 0;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }

        /* Buttons - Professional */
        .stButton>button {
            width: 100%;
            background: #2563eb;
            color: white;
            border: none;
            padding: 10px 20px;
            font-weight: 500;
            font-size: 15px;
            transition: all 0.2s ease;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            letter-spacing: 0.3px;
        }

        .stButton>button:hover {
            background: #1e40af;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
        }

        /* Sidebar - Professional */
        [data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
            color: #0f172a;
            font-weight: 600;
        }

        /* Progress Steps - Professional */
        .step-success {
            background: white;
            color: #059669;
            padding: 12px 16px;
            margin: 8px 0;
            font-weight: 500;
            border: 1px solid #d1fae5;
            border-left: 3px solid #059669;
            font-size: 14px;
        }

        .step-active {
            background: white;
            color: #2563eb;
            padding: 12px 16px;
            margin: 8px 0;
            font-weight: 600;
            border: 1px solid #bfdbfe;
            border-left: 3px solid #2563eb;
            font-size: 14px;
            box-shadow: 0 2px 6px rgba(37, 99, 235, 0.1);
        }

        .step-pending {
            background: white;
            color: #94a3b8;
            padding: 12px 16px;
            margin: 8px 0;
            font-weight: 400;
            border: 1px solid #e2e8f0;
            border-left: 3px solid #cbd5e1;
            font-size: 14px;
        }

        /* Tabs - Professional */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background: white;
            padding: 0;
            border-bottom: 2px solid #e2e8f0;
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #64748b;
            font-weight: 500;
            padding: 14px 28px;
            transition: all 0.2s ease;
            border-bottom: 2px solid transparent;
            font-size: 15px;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #2563eb;
            background: #f8fafc;
        }

        .stTabs [aria-selected="true"] {
            background: transparent;
            color: #2563eb !important;
            border-bottom: 2px solid #2563eb;
            font-weight: 600;
        }

        /* Quiz Options - Professional */
        .quiz-option {
            background: white;
            padding: 16px 20px;
            margin: 10px 0;
            border: 1px solid #e2e8f0;
            transition: all 0.2s ease;
            cursor: pointer;
        }

        .quiz-option:hover {
            border-color: #2563eb;
            background: #f8fafc;
        }

        /* Info/Warning/Error boxes - Professional */
        .stAlert {
            border-left-width: 4px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        }

        /* File Uploader - Professional */
        [data-testid="stFileUploader"] {
            background: white !important;
            padding: 20px;
            border: 2px dashed #cbd5e1;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }

        [data-testid="stFileUploader"] section {
            background: white !important;
        }

        [data-testid="stFileUploader"] > div {
            background: white !important;
        }

        [data-testid="stFileUploadDropzone"] {
            background: #f8fafc !important;
        }

        [data-testid="stFileUploader"] button {
            background: #2563eb !important;
            color: white !important;
        }

        /* Text Area - Professional */
        .stTextArea textarea {
            border: 1px solid #cbd5e1;
            font-size: 15px;
            padding: 12px;
            transition: all 0.2s ease;
        }

        .stTextArea textarea:focus {
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }

        /* Select Box - Professional */
        .stSelectbox > div > div {
            border: 1px solid #cbd5e1;
            transition: all 0.2s ease;
        }

        .stSelectbox > div > div:focus-within {
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }

        /* Metrics - Professional */
        [data-testid="stMetricValue"] {
            font-size: 1.875rem;
            font-weight: 700;
            color: #2563eb;
        }

        /* Expander - Professional */
        .streamlit-expanderHeader {
            background: #f8fafc;
            font-weight: 500;
            color: #0f172a;
            border: 1px solid #e2e8f0;
        }

        /* Audio Player */
        audio {
            width: 100%;
        }

        /* Spinner - Professional */
        .stSpinner > div {
            border-top-color: #2563eb !important;
        }

        /* Section Headers - Professional */
        .section-header {
            background: white;
            color: #0f172a;
            padding: 16px 0;
            font-size: 1.5rem;
            font-weight: 600;
            margin: 24px 0 16px 0;
            border-bottom: 2px solid #e2e8f0;
            letter-spacing: -0.5px;
        }

        /* Info Box - Professional */
        .info-box {
            background: #eff6ff;
            border-left: 3px solid #2563eb;
            padding: 16px 20px;
            margin: 12px 0;
            border: 1px solid #bfdbfe;
        }

        .info-box p {
            margin: 0;
            color: #1e40af;
            font-size: 14px;
            line-height: 1.6;
        }

        /* Success Box - Professional */
        .success-box {
            background: #f0fdf4;
            border-left: 3px solid #059669;
            padding: 16px 20px;
            margin: 12px 0;
            border: 1px solid #bbf7d0;
        }

        /* Quiz Question Box - Professional */
        .quiz-question {
            background: white;
            padding: 28px;
            border: 2px solid #2563eb;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.1);
            font-size: 1.25rem;
            font-weight: 500;
            color: #0f172a;
            text-align: center;
            line-height: 1.6;
        }

        /* Radio Buttons - Professional */
        .stRadio {
            background: transparent !important;
        }

        .stRadio > div {
            background: transparent !important;
            padding: 0;
            border: none;
        }

        .stRadio > label {
            background: transparent !important;
            color: #0f172a !important;
        }

        .stRadio [data-baseweb="radio"] {
            background: white !important;
        }

        /* File Info Box */
        .file-info {
            background: white;
            padding: 16px 20px;
            margin: 12px 0;
            border: 1px solid #e2e8f0;
        }

        /* Download Button - Professional */
        .stDownloadButton>button {
            background: #059669;
            color: white;
            border: none;
            padding: 10px 20px;
            font-weight: 500;
            font-size: 15px;
            transition: all 0.2s ease;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        .stDownloadButton>button:hover {
            background: #047857;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
        }

        /* Divider - Professional */
        hr {
            border: none;
            height: 1px;
            background: #e2e8f0;
            margin: 24px 0;
        }

        /* Custom Scrollbar - Professional */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #f1f5f9;
        }

        ::-webkit-scrollbar-thumb {
            background: #cbd5e1;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }

        /* Additional background fixes */
        [data-testid="stMarkdownContainer"] {
            background: transparent !important;
        }

        [data-testid="stText"] {
            color: #1e293b !important;
        }

        /* Input field backgrounds */
        input, textarea, select {
            background: white !important;
            color: #0f172a !important;
        }

        /* Markdown text color */
        .stMarkdown {
            color: #1e293b !important;
        }

        /* Widget containers */
        [data-testid="stForm"] {
            background: transparent !important;
        }

        /* Column containers */
        [data-testid="column"] {
            background: transparent !important;
        }

        /* Ensure metric labels are visible */
        [data-testid="stMetricLabel"] {
            color: #64748b !important;
        }

        /* Fix any remaining dark mode elements */
        [data-baseweb="base-input"] {
            background: white !important;
        }

        [data-baseweb="select"] {
            background: white !important;
        }

        /* Additional file uploader fixes */
        [data-testid="stFileUploader"] label {
            color: #0f172a !important;
        }

        [data-testid="stFileUploader"] small {
            color: #64748b !important;
        }

        /* Make sure no elements have black background */
        section, div[data-testid] {
            background-color: inherit;
        }

        /* Override any dark theme */
        @media (prefers-color-scheme: dark) {
            html, body, .stApp, .main {
                background: #f8fafc !important;
                color: #1e293b !important;
            }
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Header - Professional
    st.markdown(
        """
        <div class="main-header">
            <h1 class="main-title">German Language Learning Platform</h1>
            <p class="main-subtitle">Advanced B2-Level Vocabulary Extraction & Analysis System</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    if "transcript" not in st.session_state:
        st.session_state.transcript = ""
    if "vocabulary" not in st.session_state:
        st.session_state.vocabulary = []
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
    if "quiz_total" not in st.session_state:
        st.session_state.quiz_total = 0
    if "current_step" not in st.session_state:
        st.session_state.current_step = 1

    # Sidebar - Progress tracker
    with st.sidebar:
        st.markdown("## Processing Pipeline")

        # Progress indicators with custom styling
        steps = {
            1: "Step 1: Audio Input",
            2: "Step 2: Transcription",
            3: "Step 3: Vocabulary Extraction",
            4: "Step 4: Assessment",
        }

        for step_num, step_name in steps.items():
            if step_num < st.session_state.current_step:
                st.markdown(
                    f'<div class="step-success">✓ {step_name}</div>',
                    unsafe_allow_html=True,
                )
            elif step_num == st.session_state.current_step:
                st.markdown(
                    f'<div class="step-active">→ {step_name}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="step-pending">○ {step_name}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # Statistics
        if st.session_state.transcript:
            st.metric("Words in Transcript", len(st.session_state.transcript.split()))

        if st.session_state.vocabulary:
            st.metric("Vocabulary Count", len(st.session_state.vocabulary))

        if st.session_state.quiz_total > 0:
            accuracy = (st.session_state.quiz_score / st.session_state.quiz_total) * 100
            st.metric("Quiz Accuracy", f"{accuracy:.1f}%")

        st.markdown("---")

        if st.button("Reset Pipeline"):
            st.session_state.transcript = ""
            st.session_state.vocabulary = []
            st.session_state.quiz_score = 0
            st.session_state.quiz_total = 0
            st.session_state.current_step = 1
            st.rerun()

    # Main content
    tabs = st.tabs(["1. Audio Input", "2. Vocabulary Study", "3. Knowledge Assessment"])

    # ========================================================================
    # TAB 1: AUDIO INPUT & TRANSCRIPTION
    # ========================================================================
    with tabs[0]:
        st.markdown('<div class="section-header">Audio Input & Transcription</div>', unsafe_allow_html=True)

        input_method = st.radio(
            "Choose input method:",
            ["Upload Audio File", "Paste Text", "Load from File"],
        )

        if input_method == "Upload Audio File":
            st.markdown("#### Upload Audio File")
            st.markdown('<div class="info-box"><p>Supported formats: MP3, WAV, M4A, MPEG, MP4, WEBM (max 25MB)</p></div>', unsafe_allow_html=True)

            uploaded_audio = st.file_uploader(
                "Choose an audio file",
                type=["mp3", "wav", "m4a", "mpeg", "mp4", "webm"],
            )

            if uploaded_audio:
                st.audio(uploaded_audio)

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**File:** {uploaded_audio.name}")
                    st.write(f"**Size:** {uploaded_audio.size / 1024:.2f} KB")

                with col2:
                    if st.button("Transcribe Audio", type="primary"):
                        if uploaded_audio.size > 25 * 1024 * 1024:
                            st.error("File too large! Maximum size is 25MB.")
                        else:
                            with st.spinner(
                                "Transcribing audio... This may take a moment."
                            ):
                                transcript = transcribe_uploaded_audio(uploaded_audio)

                                if transcript:
                                    st.session_state.transcript = transcript
                                    st.session_state.current_step = 2
                                    st.success("Transcription completed successfully.")
                                    st.rerun()

        elif input_method == "Paste Text":
            st.markdown("#### Paste German Text")

            text_input = st.text_area(
                "Paste your German text here:",
                height=200,
                placeholder="Ich habe also die Mails beantwortet, solange die Kinder noch geschlafen haben...",
            )

            if st.button("Load Text", type="primary"):
                if text_input.strip():
                    st.session_state.transcript = text_input.strip()
                    st.session_state.current_step = 2
                    st.success("Text loaded successfully.")
                    st.rerun()
                else:
                    st.error("Please enter some text!")

        else:  # Load from File
            st.markdown("#### Load Text from File")

            uploaded_text = st.file_uploader("Choose a text file", type=["txt"])

            if uploaded_text:
                text_content = uploaded_text.read().decode("utf-8")
                st.text_area("File content:", text_content, height=200)

                if st.button("Load File", type="primary"):
                    st.session_state.transcript = text_content
                    st.session_state.current_step = 2
                    st.success("File loaded successfully.")
                    st.rerun()

        # Display current transcript
        if st.session_state.transcript:
            st.markdown("---")
            st.markdown("#### Transcribed Content")

            st.markdown(
                f"""
                <div class="transcript-box">
                    {st.session_state.transcript}
                </div>
            """,
                unsafe_allow_html=True,
            )

            # Extract vocabulary button
            if not st.session_state.vocabulary:
                if st.button("Extract B2 Vocabulary", type="primary"):
                    with st.spinner("Analyzing text and extracting vocabulary..."):
                        vocabulary = extract_b2_vocabulary(st.session_state.transcript)
                        st.session_state.vocabulary = vocabulary
                        st.session_state.current_step = 3

                    if vocabulary:
                        st.success(f"Successfully extracted {len(vocabulary)} B2-level vocabulary words.")
                        st.balloons()
                        st.rerun()
            else:
                st.success(
                    f"{len(st.session_state.vocabulary)} vocabulary words ready."
                )
                st.info("Navigate to the 'Vocabulary Study' tab to review extracted terms.")

    # ========================================================================
    # TAB 2: STUDY VOCABULARY
    # ========================================================================
    with tabs[1]:
        st.markdown('<div class="section-header">Vocabulary Analysis & Study</div>', unsafe_allow_html=True)

        if not st.session_state.vocabulary:
            st.warning("No vocabulary data available.")
            st.info("Please provide audio or text in Step 1, then extract vocabulary.")
        else:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>{len(st.session_state.vocabulary)}</h3>
                        <p>Total Words</p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                nouns = len(
                    [v for v in st.session_state.vocabulary if v["pos"] == "noun"]
                )
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>{nouns}</h3>
                        <p>Nouns</p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            with col3:
                verbs = len(
                    [v for v in st.session_state.vocabulary if v["pos"] == "verb"]
                )
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>{verbs}</h3>
                        <p>Verbs</p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            with col4:
                others = len(st.session_state.vocabulary) - nouns - verbs
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>{others}</h3>
                        <p>Other</p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Filter and sort options
            col1, col2 = st.columns(2)
            with col1:
                all_pos = list(set([v["pos"] for v in st.session_state.vocabulary]))
                filter_pos = st.multiselect(
                    "Filter by part of speech:", options=all_pos, default=all_pos
                )

            with col2:
                sort_by = st.selectbox(
                    "Sort by:", ["Original order", "Alphabetical", "Part of speech"]
                )

            # Filter and sort vocabulary
            filtered_vocab = [
                v for v in st.session_state.vocabulary if v["pos"] in filter_pos
            ]

            if sort_by == "Alphabetical":
                filtered_vocab = sorted(filtered_vocab, key=lambda x: x["word"])
            elif sort_by == "Part of speech":
                filtered_vocab = sorted(
                    filtered_vocab, key=lambda x: (x["pos"], x["word"])
                )

            st.markdown("---")

            # Display vocabulary cards
            for idx, word_data in enumerate(filtered_vocab, 1):
                article_text = (
                    f"{word_data.get('article', '')} "
                    if word_data.get("article")
                    else ""
                )

                st.markdown(
                    f"""
                    <div class="vocab-card">
                        <div class="word-title">
                            {idx}. {article_text}{word_data['word']}
                        </div>
                        <div class="translation">
                            {word_data['translation']}
                        </div>
                        <span class="pos-badge">{word_data['pos']}</span>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

                if word_data.get("example"):
                    st.markdown(
                        f"""
                        <div class="example">
                            <strong>Example:</strong><br>
                            {word_data['example']}
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                if word_data.get("b2_relevance"):
                    with st.expander("Why is this B2-relevant?"):
                        st.write(word_data["b2_relevance"])

                st.markdown("")

            # Export option
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Export to CSV"):
                    import pandas as pd

                    df = pd.DataFrame(st.session_state.vocabulary)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="german_b2_vocabulary.csv",
                        mime="text/csv",
                    )

            with col2:
                if st.button("Continue to Assessment"):
                    st.session_state.current_step = 4
                    st.success("Proceeding to assessment module.")
                    st.rerun()

    # ========================================================================
    # TAB 3: QUIZ MODE
    # ========================================================================
    with tabs[2]:
        st.markdown('<div class="section-header">Knowledge Assessment</div>', unsafe_allow_html=True)

        if not st.session_state.vocabulary:
            st.warning("No vocabulary data available for assessment.")
            st.info("Please complete Steps 1 and 2 first.")
        else:
            # Score display
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Correct", st.session_state.quiz_score)
            with col2:
                st.metric("Total", st.session_state.quiz_total)
            with col3:
                if st.session_state.quiz_total > 0:
                    accuracy = (
                        st.session_state.quiz_score / st.session_state.quiz_total
                    ) * 100
                    st.metric("Accuracy", f"{accuracy:.1f}%")
                else:
                    st.metric("Accuracy", "0%")

            st.markdown("---")

            # Quiz state management
            if "current_quiz" not in st.session_state:
                st.session_state.current_quiz = None
                st.session_state.quiz_answered = False
                st.session_state.selected_answer = None

            # Generate new question
            col1, col2 = st.columns([3, 1])

            with col1:
                if (
                    st.button("Generate Question", type="primary")
                    or st.session_state.current_quiz is None
                ):
                    import random

                    word_data = random.choice(st.session_state.vocabulary)

                    with st.spinner("Generating quiz question..."):
                        quiz = generate_quiz_question(word_data)

                        if quiz:
                            st.session_state.current_quiz = quiz
                            st.session_state.quiz_answered = False
                            st.session_state.selected_answer = None
                            st.rerun()

            with col2:
                if st.button("Reset Score"):
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_total = 0
                    st.rerun()

            # Display quiz question
            if st.session_state.current_quiz:
                quiz = st.session_state.current_quiz

                st.markdown('<div class="section-header">Question</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="quiz-question">{quiz["question"]}</div>', unsafe_allow_html=True)

                st.markdown("")

                # Answer options as buttons
                for i, option in enumerate(quiz["options"]):
                    button_type = "secondary"

                    if st.session_state.quiz_answered:
                        if i == quiz["correct"]:
                            button_type = "primary"  # Correct answer

                    if st.button(
                        f"{chr(65+i)}. {option}",
                        key=f"option_{i}",
                        disabled=st.session_state.quiz_answered,
                        use_container_width=True,
                    ):
                        st.session_state.selected_answer = i
                        st.session_state.quiz_answered = True
                        st.session_state.quiz_total += 1

                        if i == quiz["correct"]:
                            st.session_state.quiz_score += 1

                        st.rerun()

                # Show result
                if st.session_state.quiz_answered:
                    st.markdown("---")

                    if st.session_state.selected_answer == quiz["correct"]:
                        st.success("**Correct.** Well done!")
                    else:
                        st.error(
                            f"**Incorrect.** The correct answer is: **{quiz['options'][quiz['correct']]}**"
                        )

                    st.info(f"**Explanation:** {quiz.get('explanation', 'N/A')}")

                    st.markdown("")

                    if st.button(
                        "Next Question", type="primary", use_container_width=True
                    ):
                        st.session_state.current_quiz = None
                        st.session_state.quiz_answered = False
                        st.session_state.selected_answer = None
                        st.rerun()


if __name__ == "__main__":
    # Check for API key
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not configured.")
        st.info(
            """
        **For Deployment:**
        - Go to Streamlit Cloud → Your App → Settings → Secrets
        - Add: `OPENAI_API_KEY = "your-key-here"`

        **For Local Development:**
        - Create a `.env` file with: `OPENAI_API_KEY=your-key-here`
        """
        )

        st.markdown("---")
        st.markdown("### Demo Mode")
        st.markdown(
            """
            <div class="info-box">
                <p><strong>Preview the platform with sample data:</strong></p>
                <p>Click below to explore the interface functionality with pre-loaded vocabulary examples.</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

        if st.button("Launch Demo", type="primary"):
            # Create demo session state
            st.session_state.transcript = """Ich habe also die Mails beantwortet, solange die Kinder noch geschlafen haben.
            Dann habe ich sie in den Kindergarten gebracht und bin zum Supermarkt gegangen."""

            st.session_state.vocabulary = [
                {
                    "word": "beantworten",
                    "translation": "to answer, to reply",
                    "pos": "verb",
                    "article": "",
                    "example": "Ich beantworte die E-Mails jeden Morgen.",
                    "b2_relevance": "Important verb used in professional contexts for responding to communications."
                },
                {
                    "word": "Kindergarten",
                    "translation": "kindergarten",
                    "pos": "noun",
                    "article": "der",
                    "example": "Die Kinder gehen jeden Tag in den Kindergarten.",
                    "b2_relevance": "Common compound noun in family and education contexts."
                },
                {
                    "word": "Supermarkt",
                    "translation": "supermarket",
                    "pos": "noun",
                    "article": "der",
                    "example": "Ich kaufe Lebensmittel im Supermarkt.",
                    "b2_relevance": "Essential vocabulary for daily life and shopping contexts."
                }
            ]
            st.session_state.current_step = 3
            st.session_state.quiz_score = 0
            st.session_state.quiz_total = 0
            st.rerun()

    main()
