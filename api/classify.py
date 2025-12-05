from http.server import BaseHTTPRequestHandler
import joblib
import os
import re
import unicodedata
from api._utils import send_cors_preflight, send_json_response, send_error_response, get_request_body

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

# Load the model outside the class to leverage caching between requests (Warm start)
# Use an absolute path based on the current file's location
try:
    model_path = os.path.join(os.path.dirname(__file__), 'classify_model.pkl')
    model = joblib.load(model_path)
    print("Classification model loaded successfully.")
except Exception as e:
    print(f"Error loading classification model: {e}")
    model = None

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight request"""
        send_cors_preflight(self, 'POST, OPTIONS')

    def do_POST(self):
        """Handle POST request for transaction classification"""
        try:
            if not model:
                raise Exception("Classification model is not loaded. Check server logs.")

            # 1. Read and parse the request body
            data = get_request_body(self)
            descriptions = data.get('descriptions')

            if not descriptions:
                send_json_response(
                    self,
                    {'error': "The function must be called with 'descriptions' (a list of strings)."},
                    status_code=400,
                    methods='POST, OPTIONS'
                )
                return
            
            print(f"Classifying descriptions: {descriptions}")

            # 2. Limpa descrições
            cleaned_descriptions = []
            sujeira_prefixos = ['COMPRA', 'PGTO', 'DEBITO', 'CREDITO', 'PIX', 
                                'TED', 'DOC', 'EXTRATO', 'ELO', 'VISTA', 'Visa', 
                                'QR', 'CODE', 'DINAMICO', 'DES', 'TRANSFERENCIA',
                                'REM', 'PAGTO', 'COBRANCA', 'ESTATICO', 'ENVIADO',
                                'PAGAMENTO', 'REALIZADA', 'PICPAY', 'CARD', 'RECEBIDO',
                                'SALDO']
            sujeira_sufixos = ['SP', 'RJ', 'BH', 'CURITIBA', 'MATRIZ', 'FILIAL', 'S.A.', 'LTDA', 'PAGAMENTOS']
            
            for desc in descriptions:
                # Remove accents and dates first
                cleaned_desc = remove_accents(desc)
                cleaned_desc = re.sub(r'\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?', '', cleaned_desc)

                # Original cleaning
                cleaned_desc = ' '.join(re.findall(r'\b(?!\d+\b)\w{3,}\b', cleaned_desc)).upper().strip()
                all_sujeira = sujeira_prefixos + sujeira_sufixos
                pattern = r'\b(' + '|'.join(all_sujeira) + r')\b'
                cleaned_desc = re.sub(pattern, '', cleaned_desc)
                
                cleaned_descriptions.append(re.sub(r'\s+', ' ', cleaned_desc).strip())

            print(f"Cleaned descriptions for classification: {cleaned_descriptions}")

            # 3. Separa as descrições que serão classificadas e as que serão "OUTROS"
            to_classify = []
            to_classify_indices = []
            results = [None] * len(descriptions)

            for i, cleaned_desc in enumerate(cleaned_descriptions):
                if cleaned_desc:
                    to_classify.append(cleaned_desc)
                    to_classify_indices.append(i)
                else:
                    results[i] = {
                        "description": descriptions[i],
                        "category": "OUTROS",
                        "confidence": 1.0
                    }

            # 4. Make predictions
            if to_classify:
                predictions = model.predict(to_classify)
                try:
                    probas = model.predict_proba(to_classify).max(axis=1)
                except AttributeError:
                    probas = [1.0] * len(to_classify)

                # 5. Prepara os dados de resposta
                for i, pred_index in enumerate(to_classify_indices):
                    results[pred_index] = {
                        "description": descriptions[pred_index],
                        "category": predictions[i],
                        "confidence": float(probas[i])
                    }
            
            response_data = {"results": results}
            
            print(f"Prediction successful: {response_data}")

            # 6. Send the response back to the client
            send_json_response(self, {'data': response_data}, methods='POST, OPTIONS')

        except Exception as error:
            print(f"Error classifying transaction: {error}")
            send_error_response(self, error, methods='POST, OPTIONS')