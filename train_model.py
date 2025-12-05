import pandas as pd
import numpy as np
import random
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

def train_and_save_model():
    # Abre base de dados
    with open("base_dados.json", "r") as json_file:
        base_dados = json.load(json_file)

    categorias = list(base_dados.keys())

    # Sujeira para as transacoes
    sujeira_prefixos = ['COMPRA', 'PGTO', 'DEBITO', 'CREDITO', 'PIX', 'TED', 'DOC', 'EXTRATO', 'COMPRA ELO', 'COMPRA VISA', 'ELO', 'VISTA']
    sujeira_sufixos = ['SP', 'RJ', 'BH', 'CURITIBA', 'MATRIZ', 'FILIAL', 'S.A.', 'LTDA', 'PAGAMENTOS']

    def gerar_dataset(qtd_linhas=5000):
        transacoes = []

        for _ in range(qtd_linhas):
            # Escolhe uma loja aleatória
            cat = random.choice(categorias)
            loja = random.choice(base_dados[cat])

            # Gera a descrição suja
            desc = loja
            if random.random() < 0.6: desc = f"{random.choice(sujeira_prefixos)} {desc}"
            if random.random() < 0.4: desc = f"{desc} {random.choice(sujeira_sufixos)}"
            if random.random() < 0.3: desc = f"{desc} {random.randint(10, 9999)}"

            transacoes.append([desc, cat])

        return pd.DataFrame(transacoes, columns=['Descricao', 'Categoria'])

    # Executa e salva
    print("Gerando 10.000 transações...")
    df = gerar_dataset(10000)

    # Limpa descrições
    df['Descricao'] = df['Descricao'].str.findall(r'\b(?!\d+\b)\w{2,}\b').str.join(' ').str.upper().str.strip()
    for w in sujeira_prefixos + sujeira_sufixos:
        df['Descricao'] = df['Descricao'].str.replace(w, '')
    df['Descricao'] = df['Descricao'].str.replace(r'\s+', ' ', regex=True).str.strip()

    df['Categoria'] = df['Categoria'].str.upper().str.strip()

    X = df['Descricao']
    y = df['Categoria']

    model = make_pipeline(
        TfidfVectorizer(),
        RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    )

    print("Treinando modelos para todas as colunas...")
    model.fit(X, y)
    print("Treino concluído!")

    # Salva o modelo
    joblib.dump(model, 'api/classify_model.pkl')
    print("Modelo salvo em api/classify_model.pkl")

if __name__ == '__main__':
    train_and_save_model()
