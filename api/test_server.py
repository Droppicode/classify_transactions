from http.server import HTTPServer
from api.classifyTransactions import handler

def run(server_class=HTTPServer, handler_class=handler, port=8002):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting test server for classifyTransactions on http://localhost:{port}...')
    httpd.serve_forever()

if __name__ == "__main__":
    run()
