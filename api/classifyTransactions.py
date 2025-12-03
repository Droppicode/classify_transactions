from http.server import BaseHTTPRequestHandler
import joblib
import os
from api._utils import send_cors_preflight, send_json_response, send_error_response, get_request_body

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
            single_description = data.get('description')

            if descriptions:
                print(f"Classifying descriptions: {descriptions}")
                to_classify = descriptions
            elif single_description:
                print(f"Classifying description: '{single_description}'")
                to_classify = [single_description]
            else:
                send_json_response(
                    self,
                    {'error': "The function must be called with 'descriptions' (a list of strings) or 'description' (a single string)."},
                    status_code=400,
                    methods='POST, OPTIONS'
                )
                return

            # 2. Make the predictions
            predictions = model.predict(to_classify)
            
            # Try to get confidence scores
            try:
                probas = model.predict_proba(to_classify).max(axis=1)
            except AttributeError:
                probas = [1.0] * len(to_classify)

            # 3. Prepare the response data
            results = []
            for i, description in enumerate(to_classify):
                results.append({
                    "description": description,
                    "category": predictions[i],
                    "confidence": float(probas[i])
                })
            
            response_data = {"results": results}
            
            print(f"Prediction successful: {response_data}")

            # 4. Send the response back to the client
            send_json_response(self, {'data': response_data}, methods='POST, OPTIONS')

        except Exception as error:
            print(f"Error classifying transaction: {error}")
            send_error_response(self, error, methods='POST, OPTIONS')