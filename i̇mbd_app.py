import gradio as gr
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb

# Model ve kelime indeksi
model = load_model("best_model.h5")
word_index = imdb.get_word_index()
maxlen = 200

def text_to_sequence(text):
    tokens = text.lower().split()
    seq = [word_index.get(word, 2) for word in tokens]  # 2 = <UNK>
    return pad_sequences([seq], maxlen=maxlen)

def predict_sentiment(review):
    seq = text_to_sequence(review)
    prob = float(model.predict(seq)[0][0])
    
    # Kategori ve renk belirleme
    if prob >= 0.8:
        sentiment = "Positive 😃"
        color = "green"
    elif 0.6 <= prob < 0.8:
        sentiment = "Neutral 🤔"
        color = "orange"
    else:
        sentiment = "Negative 😞"
        color = "red"
    
    # HTML ile çıktı
    html_output = f"""
    <div style="font-size:20px; font-family:sans-serif;">
        <p>🎬 <b>Tahmin:</b> <span style="color:{color}">{sentiment}</span></p>
        <p>💡 <b>Olabilirlik:</b> {prob:.2f}</p>
        <hr>
        <p style="font-size:14px; color:gray;">Not: 0.60–0.79 arası nötr, 0.80 üstü pozitif, 0.59 altı negatif olarak sınıflandırılır.</p>
    </div>
    """
    return html_output

# Gradio arayüzü
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, placeholder="Film yorumunu buraya yazın..."),
    outputs=gr.HTML(),
    title="IMDb Sentiment Analyzer",
    description="Film yorumunu girin ve modelin duygu tahminini görsel olarak görün."
)

iface.launch(share=True)
