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
        page_title="German B2 Learning Pipeline", page_icon="🇩🇪", layout="wide"
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        .vocab-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 5px solid #1f77b4;
        }
        .word-title {
            font-size: 28px;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 5px;
        }
        .translation {
            font-size: 20px;
            color: #666;
            font-style: italic;
            margin-bottom: 15px;
        }
        .example {
            background-color: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #1f77b4;
            margin: 10px 0;
            border-radius: 5px;
        }
        .transcript-box {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #dee2e6;
            margin: 20px 0;
        }
        .stButton>button {
            width: 100%;
        }
        .metric-card {
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Header
    st.title("🇩🇪 Complete German B2 Learning Pipeline")
    st.markdown("### Audio → Transcription → Vocabulary → Learning")

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
        st.header("📊 Pipeline Progress")

        # Progress indicators
        steps = {
            1: "🎤 Audio Input",
            2: "📝 Transcription",
            3: "📚 Vocabulary",
            4: "🎯 Learning",
        }

        for step_num, step_name in steps.items():
            if step_num < st.session_state.current_step:
                st.success(f"✅ {step_name}")
            elif step_num == st.session_state.current_step:
                st.info(f"▶️ {step_name}")
            else:
                st.text(f"⏸️ {step_name}")

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

        if st.button("🔄 Reset Pipeline"):
            st.session_state.transcript = ""
            st.session_state.vocabulary = []
            st.session_state.quiz_score = 0
            st.session_state.quiz_total = 0
            st.session_state.current_step = 1
            st.rerun()

    # Main content
    tabs = st.tabs(["🎤 Step 1: Audio Input", "📚 Step 2: Study", "🎯 Step 3: Quiz"])

    # ========================================================================
    # TAB 1: AUDIO INPUT & TRANSCRIPTION
    # ========================================================================
    with tabs[0]:
        st.header("Step 1: Provide German Audio or Text")

        input_method = st.radio(
            "Choose input method:",
            ["Upload Audio File", "Paste Text", "Load from File"],
        )

        if input_method == "Upload Audio File":
            st.subheader("🎤 Upload Audio File")
            st.info("Supported formats: MP3, WAV, M4A, MPEG, MP4, WEBM (max 25MB)")

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
                    if st.button("🎙️ Transcribe", type="primary"):
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
                                    st.success("✅ Transcription complete!")
                                    st.rerun()

        elif input_method == "Paste Text":
            st.subheader("📝 Paste German Text")

            text_input = st.text_area(
                "Paste your German text here:",
                height=200,
                placeholder="Ich habe also die Mails beantwortet, solange die Kinder noch geschlafen haben...",
            )

            if st.button("✅ Use This Text", type="primary"):
                if text_input.strip():
                    st.session_state.transcript = text_input.strip()
                    st.session_state.current_step = 2
                    st.success("✅ Text loaded!")
                    st.rerun()
                else:
                    st.error("Please enter some text!")

        else:  # Load from File
            st.subheader("📁 Load Text from File")

            uploaded_text = st.file_uploader("Choose a text file", type=["txt"])

            if uploaded_text:
                text_content = uploaded_text.read().decode("utf-8")
                st.text_area("File content:", text_content, height=200)

                if st.button("✅ Use This File", type="primary"):
                    st.session_state.transcript = text_content
                    st.session_state.current_step = 2
                    st.success("✅ File loaded!")
                    st.rerun()

        # Display current transcript
        if st.session_state.transcript:
            st.markdown("---")
            st.subheader("📄 Current Transcript")

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
                if st.button("📚 Extract B2 Vocabulary", type="primary"):
                    with st.spinner("Analyzing text and extracting vocabulary..."):
                        vocabulary = extract_b2_vocabulary(st.session_state.transcript)
                        st.session_state.vocabulary = vocabulary
                        st.session_state.current_step = 3

                    if vocabulary:
                        st.success(f"✅ Extracted {len(vocabulary)} B2-level words!")
                        st.balloons()
                        st.rerun()
            else:
                st.success(
                    f"✅ {len(st.session_state.vocabulary)} vocabulary words ready!"
                )
                st.info("👉 Go to the 'Study' tab to review vocabulary")

    # ========================================================================
    # TAB 2: STUDY VOCABULARY
    # ========================================================================
    with tabs[1]:
        st.header("Step 2: Study Vocabulary")

        if not st.session_state.vocabulary:
            st.warning("⚠️ No vocabulary available yet!")
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
                        <p><strong>Part of speech:</strong> {word_data['pos']}</p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

                if word_data.get("example"):
                    st.markdown(
                        f"""
                        <div class="example">
                            <strong>📝 Example:</strong><br>
                            {word_data['example']}
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                if word_data.get("b2_relevance"):
                    with st.expander("ℹ️ Why is this B2-relevant?"):
                        st.write(word_data["b2_relevance"])

                st.markdown("")

            # Export option
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📥 Export to CSV"):
                    import pandas as pd

                    df = pd.DataFrame(st.session_state.vocabulary)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv,
                        file_name="german_b2_vocabulary.csv",
                        mime="text/csv",
                    )

            with col2:
                if st.button("🎯 Ready for Quiz!"):
                    st.session_state.current_step = 4
                    st.success("✅ Moving to quiz mode!")
                    st.rerun()

    # ========================================================================
    # TAB 3: QUIZ MODE
    # ========================================================================
    with tabs[2]:
        st.header("Step 3: Test Your Knowledge")

        if not st.session_state.vocabulary:
            st.warning("⚠️ No vocabulary available for quiz!")
            st.info("Please complete Steps 1 and 2 first.")
        else:
            # Score display
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("✅ Correct", st.session_state.quiz_score)
            with col2:
                st.metric("📊 Total", st.session_state.quiz_total)
            with col3:
                if st.session_state.quiz_total > 0:
                    accuracy = (
                        st.session_state.quiz_score / st.session_state.quiz_total
                    ) * 100
                    st.metric("🎯 Accuracy", f"{accuracy:.1f}%")
                else:
                    st.metric("🎯 Accuracy", "0%")

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
                    st.button("🎲 New Question", type="primary")
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
                if st.button("🔄 Reset Score"):
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_total = 0
                    st.rerun()

            # Display quiz question
            if st.session_state.current_quiz:
                quiz = st.session_state.current_quiz

                st.markdown("### 📝 Complete the sentence:")
                st.markdown(f"## {quiz['question']}")

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
                        st.success("🎉 **Correct!** Well done!")
                    else:
                        st.error(
                            f"❌ **Wrong!** The correct answer is: **{quiz['options'][quiz['correct']]}**"
                        )

                    st.info(f"**💡 Explanation:** {quiz.get('explanation', 'N/A')}")

                    st.markdown("")

                    if st.button(
                        "➡️ Next Question", type="primary", use_container_width=True
                    ):
                        st.session_state.current_quiz = None
                        st.session_state.quiz_answered = False
                        st.session_state.selected_answer = None
                        st.rerun()


if __name__ == "__main__":
    # Check for API key
    if not OPENAI_API_KEY:
        st.error("⚠️ OpenAI API key not found!")
        st.info(
            """
        Please add your OpenAI API key to the `.env` file:
        
        ```
        OPENAI_API_KEY=your-api-key-here
        ```
        """
        )
        st.stop()

    main()
