"""
Shared utility functions for Vercel serverless functions
"""
import json

def send_cors_headers(handler, methods='GET, POST, OPTIONS'):
    """Send CORS headers"""
    handler.send_header('Access-Control-Allow-Origin', '*')
    handler.send_header('Access-Control-Allow-Methods', methods)
    handler.send_header('Access-Control-Allow-Headers', 'Content-Type')

def send_json_response(handler, data, status_code=200, methods='GET, POST, OPTIONS'):
    """Send a JSON response with proper headers"""
    handler.send_response(status_code)
    send_cors_headers(handler, methods)
    handler.send_header('Content-Type', 'application/json')
    handler.end_headers()
    handler.wfile.write(json.dumps(data).encode())

def send_text_response(handler, text, status_code=200, methods='GET, POST, OPTIONS'):
    """Send a text response with proper headers"""
    handler.send_response(status_code)
    send_cors_headers(handler, methods)
    handler.send_header('Content-Type', 'text/plain; charset=utf-8')
    handler.end_headers()
    handler.wfile.write(text.encode('utf-8'))

def send_error_response(handler, error, status_code=500, methods='GET, POST, OPTIONS'):
    """Send an error response"""
    error_data = {
        'error': str(error),
        'details': str(error.__cause__) if error.__cause__ else None
    }
    send_json_response(handler, error_data, status_code, methods)

def send_cors_preflight(handler, methods='GET, POST, OPTIONS'):
    """Handle CORS preflight OPTIONS request"""
    handler.send_response(200)
    send_cors_headers(handler, methods)
    handler.end_headers()

def get_request_body(handler):
    """Read and parse JSON request body"""
    content_length = int(handler.headers.get('Content-Length', 0))
    if content_length == 0:
        return {}
    body = handler.rfile.read(content_length).decode('utf-8')
    return json.loads(body) if body else {}
