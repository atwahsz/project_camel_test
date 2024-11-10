import streamlit as st
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.tokenizers.sent import simple_sentence_tokenize
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.disambiguator import Disambiguator
from camel_tools.ner import NERecognizer
from camel_tools.sentiment import SentimentAnalyzer
from camel_tools.dialect import DialectIdentifier
from camel_tools.embeddings import Embedding
from docx import Document
import io

# Load CAMeL Tools resources
morph_db = MorphologyDB.builtin_db()
morph_analyzer = Analyzer(morph_db)
disambiguator = Disambiguator.pretrained()
ner = NERecognizer.pretrained()
sentiment_analyzer = SentimentAnalyzer.pretrained()
dialect_id = DialectIdentifier.pretrained()
embedding = Embedding.pretrained()

def load_word_file(file):
    doc = Document(io.BytesIO(file.read()))
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text

def process_text(text, tokenize_words, tokenize_sentences, morph_analysis, disambiguation, ner_analysis, sentiment_analysis, dialect_identification, embeddings):
    results = {}

    # Word Tokenization
    if tokenize_words:
        results['تجزئة الكلمات'] = simple_word_tokenize(text)

    # Sentence Tokenization
    if tokenize_sentences:
        results['تجزئة الجمل'] = simple_sentence_tokenize(text)

    # Morphological Analysis
    if morph_analysis:
        results['التحليل الصرفي'] = [morph_analyzer.analyze(word) for word in results.get('تجزئة الكلمات', simple_word_tokenize(text))]

    # Morphological Disambiguation
    if disambiguation:
        results['التحليل الصرفي مع فضّ الالتباس'] = disambiguator.disambiguate(results.get('تجزئة الكلمات', simple_word_tokenize(text)))

    # Named Entity Recognition
    if ner_analysis:
        results['التعرف على الكيانات'] = ner.predict(results.get('تجزئة الكلمات', simple_word_tokenize(text)))

    # Sentiment Analysis
    if sentiment_analysis:
        results['تحليل المشاعر'] = sentiment_analyzer.predict(text)

    # Dialect Identification
    if dialect_identification:
        results['التعرف على اللهجة'] = dialect_id.predict(text)

    # Embeddings for Topic Modeling
    if embeddings:
        results['التضمين (Embeddings)'] = [embedding.embed_word(word) for word in results.get('تجزئة الكلمات', simple_word_tokenize(text))]

    return results

# Streamlit App Interface
st.title("معالجة اللغة العربية باستخدام أدوات CAMeL")
st.write("قم بتحميل مستند Word واختيار المهام التي ترغب في تطبيقها.")

uploaded_file = st.file_uploader("تحميل ملف Word", type=["docx"])
if uploaded_file:
    text = load_word_file(uploaded_file)
    st.write("النص المستخرج:")
    st.write(text)

    st.subheader("اختر المهام المطلوبة")
    tokenize_words = st.checkbox("تجزئة النص إلى كلمات")
    tokenize_sentences = st.checkbox("تجزئة النص إلى جمل")
    morph_analysis = st.checkbox("التحليل الصرفي")
    disambiguation = st.checkbox("التحليل الصرفي مع فضّ الالتباس")
    ner_analysis = st.checkbox("التعرف على الكيانات المسماة")
    sentiment_analysis = st.checkbox("تحليل المشاعر")
    dialect_identification = st.checkbox("التعرف على اللهجة")
    embeddings = st.checkbox("التضمين (Embeddings)")

    if st.button("تحليل"):
        if any([tokenize_words, tokenize_sentences, morph_analysis, disambiguation, ner_analysis, sentiment_analysis, dialect_identification, embeddings]):
            with st.spinner("جاري المعالجة..."):
                results = process_text(
                    text, 
                    tokenize_words, 
                    tokenize_sentences, 
                    morph_analysis, 
                    disambiguation, 
                    ner_analysis, 
                    sentiment_analysis, 
                    dialect_identification, 
                    embeddings
                )
            st.success("اكتملت المعالجة!")

            # Display Results
            for task, result in results.items():
                st.subheader(task)
                st.write(result)
        else:
            st.warning("يرجى اختيار مهمة واحدة على الأقل للتنفيذ.")
