import pandas as pd
import numpy as np
import random
import json
import joblib
import unicodedata
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def train_and_save_model():
    # Abre base de dados
    with open("base_dados.json", "r") as json_file:
        base_dados = json.load(json_file)

    categorias = list(base_dados.keys())

    # Sujeira para as transacoes
    sujeira_prefixos = ['COMPRA', 'PGTO', 'DEBITO', 'CREDITO', 'PIX', 
                        'TED', 'DOC', 'EXTRATO', 'ELO', 'VISTA', 'Visa', 
                        'QR', 'CODE', 'DINAMICO', 'DES', 'TRANSFERENCIA',
                        'REM', 'PAGTO', 'COBRANCA', 'ESTATICO', 'ENVIADO',
                        'PAGAMENTO', 'REALIZADA', 'PICPAY', 'CARD', 'RECEBIDO',
                        'SALDO']
    sujeira_sufixos = ['SP', 'RJ', 'BH', 'CURITIBA', 'MATRIZ', 'FILIAL', 'S.A.', 'LTDA', 'PAGAMENTOS']

    def gerar_dataset(qtd_linhas=5000):
        transacoes = []
        
        def gerar_transacao_suja(loja, cat):
            desc = loja
            if random.random() < 0.6: desc = f"{random.choice(sujeira_prefixos)} {desc}"
            if random.random() < 0.4: desc = f"{desc} {random.choice(sujeira_sufixos)}"
            if random.random() < 0.3: desc = f"{desc} {random.randint(10, 9999)}"
            return [desc, cat]

        # 1. Garante que cada item de cada categoria esteja presente pelo menos uma vez
        for cat in categorias:
            for loja in base_dados[cat]:
                transacoes.append(gerar_transacao_suja(loja, cat))

        # 2. Equilibra as categorias com as linhas restantes
        linhas_restantes = qtd_linhas - len(transacoes)
        if linhas_restantes > 0:
            linhas_por_cat = linhas_restantes // len(categorias)
            for cat in categorias:
                for _ in range(linhas_por_cat):
                    loja = random.choice(base_dados[cat])
                    transacoes.append(gerar_transacao_suja(loja, cat))
        
        # Embaralha o dataset final
        random.shuffle(transacoes)
        
        return pd.DataFrame(transacoes, columns=['Descricao', 'Categoria'])

    # Executa e salva
    print("Gerando 10.000 transações...")
    df = gerar_dataset(10000)

    # Limpa descrições
    df['Descricao'] = df['Descricao'].apply(remove_accents)
    df['Descricao'] = df['Descricao'].str.replace(r'\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?', '', regex=True)
    df['Descricao'] = df['Descricao'].str.findall(r'\b(?!\d+\b)\w{3,}\b').str.join(' ').str.upper().str.strip()
    all_sujeira = sujeira_prefixos + sujeira_sufixos
    pattern = r'\b(' + '|'.join(all_sujeira) + r')\b'
    df['Descricao'] = df['Descricao'].str.replace(pattern, '', regex=True)
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
