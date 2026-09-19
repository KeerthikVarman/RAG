import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    print(f"Starting FastAPI server at http://{host}:{port} (Swagger docs: http://localhost:{port}/docs)")
    uvicorn.run("api:app", host=host, port=port, reload=True)
