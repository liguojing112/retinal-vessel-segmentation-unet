"""Flask 应用组装入口。"""

import os

from flask import Flask, render_template

from .config import MAX_CONTENT_LENGTH, SERVER_DEBUG, SERVER_HOST, SERVER_PORT, UPLOAD_FOLDER
from .model_loader import load_model
from .routes_predict import create_predict_blueprint
from .service import PredictionService


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model_bundle = load_model()
prediction_service = PredictionService(app, model_bundle)
app.register_blueprint(create_predict_blueprint(prediction_service))


@app.route('/')
def index():
    return render_template('index.html')


def run():
    print('=' * 60)
    print('视网膜血管分割演示系统')
    print(
        f"当前模型性能: Dice {model_bundle.model_dice:.1f}% | 敏感{model_bundle.model_sens:.1f}% | 特异{model_bundle.model_spec:.1f}%"
    )
    print(f'访问地址: http://localhost:{SERVER_PORT}')
    print('=' * 60)
    app.run(debug=SERVER_DEBUG, port=SERVER_PORT, host=SERVER_HOST)


if __name__ == '__main__':
    run()
