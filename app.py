from flask import Flask, request, jsonify
from flask_cors import CORS
from models import process_pdf_query, handle_chat_query
import os
import logging

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', 
                    format='%(asctime)s %(levelname)s:%(message)s')

# Set the absolute path for the uploads directory
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    """Endpoint to upload a PDF file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file.content_type != 'application/pdf':
        return jsonify({'error': 'Invalid file type. Please upload a PDF file.'}), 400

    # Use the original filename of the uploaded PDF
    pdf_filename = file.filename
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)

    # Save the uploaded file
    file.save(pdf_path)

    app.logger.debug(f"Uploaded PDF saved at: {pdf_path}")

    return jsonify({'pdf_path': pdf_path, 'message': "Your PDF has been uploaded and processed."}), 200

@app.route('/api/pdf-query', methods=['POST'])
def pdf_query():
    """Endpoint to query the PDF content."""
    data = request.json
    app.logger.debug(f"Received query data: {data}")
    
    question = data.get('message')
    pdf_path = data.get('pdf_path')

    if question and pdf_path:
        try:
            response = process_pdf_query(pdf_path, question)
            
            # Check if response contains an error
            if 'error' in response:
                app.logger.error(f"Error from PDF query: {response['error']}")
                return jsonify(response), 500

            app.logger.debug(f"Response from PDF query: {response}")
            return jsonify({'text': response}), 200

        except Exception as e:
            app.logger.error(f"Error processing PDF query: {str(e)}")
            return jsonify({"error": "Failed to process PDF query. " + str(e)}), 500
    else:
        app.logger.error(f"Invalid input: question={question}, pdf_path={pdf_path}")
        return jsonify({"error": "Invalid input. Please provide both question and pdf_path."}), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint to handle chat messages."""
    data = request.json
    user_input = data.get('message')
    
    if user_input:
        try:
            response = handle_chat_query(user_input)
            
            # Check if response contains an error
            if 'error' in response:
                app.logger.error(f"Error from chat handler: {response['error']}")
                return jsonify(response), 500

            app.logger.debug(f"Chat response: {response}")
            return jsonify({'text': response}), 200

        except Exception as e:
            app.logger.error(f"Error during chat handling: {str(e)}")
            return jsonify({"error": "An error occurred while processing your chat input. " + str(e)}), 500
    else:
        app.logger.error("No input provided in chat request.")
        return jsonify({"error": "No input provided. Please send a message."}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
