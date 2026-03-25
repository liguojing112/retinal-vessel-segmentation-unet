import uvicorn


if __name__ == "__main__":
    print("=" * 60)
    print("视网膜血管分割演示系统")
    print("访问地址: http://localhost:5000")
    print("生产建议: gunicorn -k uvicorn.workers.UvicornWorker app.main:app -w 2 -b 0.0.0.0:5000 --timeout 120")
    print("=" * 60)
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=False)
