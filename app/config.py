"""应用配置。"""

import os

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', str(80 * 1024 * 1024)))
MAX_FILES_PER_REQUEST = int(os.getenv('MAX_FILES_PER_REQUEST', '20'))
DEFAULT_MODEL_DICE = float(os.getenv('DEFAULT_MODEL_DICE', '86.5'))
DEFAULT_MODEL_SENS = float(os.getenv('DEFAULT_MODEL_SENS', '82.0'))
DEFAULT_MODEL_SPEC = float(os.getenv('DEFAULT_MODEL_SPEC', '97.5'))
PREDICTION_THRESHOLD = float(os.getenv('PREDICTION_THRESHOLD', '0.5'))
MODEL_INPUT_SIZE = int(os.getenv('MODEL_INPUT_SIZE', '512'))
CLAHE_CLIP_LIMIT = float(os.getenv('CLAHE_CLIP_LIMIT', '2.0'))
CLAHE_TILE_GRID_SIZE = (
    int(os.getenv('CLAHE_TILE_GRID_WIDTH', '8')),
    int(os.getenv('CLAHE_TILE_GRID_HEIGHT', '8')),
)
PREPROCESS_GAMMA = float(os.getenv('PREPROCESS_GAMMA', '1.2'))
SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
SERVER_PORT = int(os.getenv('SERVER_PORT', '5000'))
SERVER_DEBUG = os.getenv('SERVER_DEBUG', 'true').lower() == 'true'
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.ppm', '.pgm', '.pbm'}
CHECKPOINT_CANDIDATES = [
    'checkpoints/best_model_800.pth',
    'checkpoints/best_model.pth',
    'checkpoints/final_model.pth',
]
