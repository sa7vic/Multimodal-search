from flask import Flask
from config import FLASK_DEBUG, MAX_UPLOAD_BYTES


def create_app():
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["DEBUG"] = FLASK_DEBUG

    from app.routes import bp
    app.register_blueprint(bp)

    return app