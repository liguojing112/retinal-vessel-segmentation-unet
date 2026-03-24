"""兼容旧入口，转发到 app.main。"""

from app.main import app, run


if __name__ == '__main__':
    run()
