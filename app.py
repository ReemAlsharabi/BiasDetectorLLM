from flask import Flask, render_template, request
import torch
from model import SimpleGPT2SequenceClassifier
from preprocessing import preprocess_text

app = Flask(__name__)

def load_model():
    model = SimpleGPT2SequenceClassifier(hidden_size=768, num_classes=2, max_seq_len=128, gpt_model_name="gpt2")
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def classify_text(text):
    input_ids, attention_mask = preprocess_text(text)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs.squeeze()
        if torch.argmax(logits).item() == 1:
            result = "Biased"
        else:
            result = "Unbiased"
    return result

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        text = request.form['text']
        prediction = classify_text(text)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
