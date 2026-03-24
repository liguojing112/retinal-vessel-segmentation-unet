"""预测路由。"""

from flask import Blueprint, jsonify, request


def create_predict_blueprint(prediction_service):
    bp = Blueprint('predict', __name__)

    @bp.route('/predict', methods=['POST'])
    def predict():
        payload, status = prediction_service.predict_files(request.files)
        return jsonify(payload), status

    return bp
