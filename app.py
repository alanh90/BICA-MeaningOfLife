from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from meaningoflife import MeaningOfLife

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize MeaningOfLife class
meaning = MeaningOfLife(api_key=os.getenv("OPENAI_API_KEY"))


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    """Process user questions and return AI responses."""
    if not request.json or 'question' not in request.json:
        return jsonify({"error": "No question provided"}), 400

    question = request.json['question']
    response = meaning.process_question(question)

    # Get the current state of future simulations for the frontend
    state = meaning.get_state()

    return jsonify({
        "response": response,
        "identity": state["identity"],
        "futures": state["futures"][:5],  # Limit to top 5 futures
        "preferred_futures": state["preferred_futures"]
    })


@app.route('/state')
def state():
    """Return the current system state for visualization."""
    state = meaning.get_state()
    return jsonify(state)


@app.route('/randomize', methods=['POST'])
def randomize():
    """Randomize the AI's identity values."""
    new_identity = meaning.randomize_identity()
    return jsonify({
        "status": "identity randomized",
        "identity": new_identity
    })


@app.route('/reset', methods=['POST'])
def reset():
    """Reset the AI's state."""
    meaning.reset()
    return jsonify({"status": "reset successful"})


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG', 'True').lower() == 'true')