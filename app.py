import pickle
from gensim.models import KeyedVectors
from flask import Flask, request, render_template

from utils import tokenizador, combinacao_de_vetores_por_soma

# Inicializa a aplicação Flask
app = Flask(__name__, template_folder="templates")

# Carregar os modelos 
w2v_dir = "models/modelo_skipgram.txt"
classificador_dir = "models/rl_sg.pkl"

print("🔄 Carregando modelos...")
w2v_modelo = KeyedVectors.load_word2vec_format(w2v_dir)
with open(classificador_dir, "rb") as f:
    classificador = pickle.load(f)
print("✅ Modelos carregados com sucesso!")


# === Rotas ===
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Obtém o texto enviado pelo formulário
    titulo = next(request.form.values())

    # Pré-processamento do título
    titulo_tokens = tokenizador(titulo)
    titulo_vetor = combinacao_de_vetores_por_soma(titulo_tokens, w2v_modelo)

    # Predição da categoria
    titulo_categoria = classificador.predict(titulo_vetor)
    output = titulo_categoria[0].capitalize()

    # Retorna a resposta renderizada
    return render_template(
        'index.html',
        title=f"Título: {titulo}",
        category=f"Categoria: {output}"
    )

if __name__ == "__main__":
    app.run(debug=True)