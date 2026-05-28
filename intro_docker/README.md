# Docker — Introduction

Docker packages applications into **containers** — lightweight, isolated environments that run the same way on any machine.

## Why Docker for ML?

Docker ensures:

- **Consistency** — your app behaves the same on your laptop, a teammate's machine, or a server
- **No dependency hell** — all requirements are frozen inside the container image
- **Easy deployment** — one command starts everything: `docker compose up`

## Key Concepts

| Concept                | What it is                                                                                            |
| ---------------------- | ----------------------------------------------------------------------------------------------------- |
| **Image**              | A read-only blueprint — contains the OS, Python, libraries, and your code. Built from a `Dockerfile`. |
| **Container**          | A running instance of an image. You can start, stop, and delete containers.                           |
| **Dockerfile**         | A recipe that describes how to build an image.                                                        |
| **docker-compose.yml** | A config file that defines one or more containers to run together.                                    |
| **Port mapping**       | Maps a port inside the container (`8000`) to a port on your machine (`8000`).                         |

## Files in This Directory

```
intro_docker/
├── app.py               # FastAPI demo app (same as hidden_repository)
├── Dockerfile            # Builds the image
├── docker-compose.yml    # Runs the container
├── .dockerignore         # Files to exclude from the image
├── requirements.txt
└── README.md
```

## 1. Build the Image

```bash
docker build -t fastapi-intro .
```

- `-t fastapi-intro` — tag (name) for the image
- `.` — build context (current directory is sent to Docker daemon)

Docker reads `Dockerfile` step by step:

```
FROM python:3.11-slim          → base image (official Python on Debian slim)
WORKDIR /app                    → all commands run inside /app
COPY requirements.txt .         → copy dependency file
RUN pip install -r ...          → install packages
COPY . .                        → copy your source code
EXPOSE 8000                     → document the port (informational)
CMD ["uvicorn", "app:app"...]   → default command when container starts
```

## 2. Run the Container

```bash
docker run -p 8000:8000 fastapi-intro
```

- `-p 8000:8000` — maps host port 8000 to container port 8000
- `fastapi-intro` — the image to run

Open **http://127.0.0.1:8000/docs** — the same API you saw before, now running inside a container.

To run in the background:

```bash
docker run -d -p 8000:8000 --name my-api fastapi-intro
```

Stop with `docker stop my-api` and remove with `docker rm my-api`.

## 3. Use Docker Compose

`docker-compose.yml` defines the service:

```yaml
services:
  app:
    build: . # build from Dockerfile in current dir
    container_name: fastapi-intro
    ports:
      - "8000:8000" # host:container port mapping
```

One command does everything:

```bash
docker compose up
```

- Builds the image (if not already built)
- Starts the container
- Shows live logs

Add `-d` for detached mode (background). Stop with `docker compose down`.

## Essential Docker Commands

| Command                              | What it does                               |
| ------------------------------------ | ------------------------------------------ |
| `docker build -t name .`             | Build an image from a Dockerfile           |
| `docker images`                      | List all images                            |
| `docker run -p HOST:CONTAINER image` | Start a container                          |
| `docker ps`                          | List running containers (`-a` shows all)   |
| `docker stop name`                   | Stop a running container                   |
| `docker rm name`                     | Remove a stopped container                 |
| `docker rmi image`                   | Remove an image                            |
| `docker compose up`                  | Build and start services from compose file |
| `docker compose down`                | Stop and remove services                   |
| `docker logs name`                   | View logs from a container                 |
| `docker exec -it name bash`          | Open a shell inside a running container    |

## The Dockerfile — Annotated

```dockerfile
FROM python:3.11-slim                # (1) Start from a small Python base image
WORKDIR /app                         # (2) Set working directory inside the container
COPY requirements.txt .              # (3) Copy only requirements first (caching trick)
RUN pip install --no-cache-dir -r requirements.txt   # (4) Install Python packages
COPY . .                             # (5) Copy the rest of the source code
EXPOSE 8000                          # (6) Tell Docker the app listens on port 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]  # (7) Start the server
```

Steps 3-4 are ordered this way so Docker can cache the `pip install` layer. If you don't change `requirements.txt`, that step is reused on rebuild — much faster.

## How This Connects to Course Projects

In your ML project APIs, you will see:

```dockerfile
# breast_cancer_copy/Dockerfile (simplified)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

And `docker-compose.yml` will have **multiple services** — your API plus MLflow:

```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v3.12.0
    ports: ["5000:5000"]
  app:
    build: .
    ports: ["8000:8000"]
    depends_on: [mlflow]
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
```

The same patterns you just learned, just with more services.

## Next Steps

1. Run `docker build -t fastapi-intro .` and `docker run -p 8000:8000 fastapi-intro`
2. Visit `http://127.0.0.1:8000/docs` and test the API running inside Docker
3. Edit `app.py` and rebuild the image — see your changes in a container
4. Try `docker compose up` instead of `docker build` + `docker run`
5. Open a shell in the running container: `docker exec -it fastapi-intro bash`
6. Look at the `breast_cancer_copy/docker-compose.yml` to see the full 2-service pattern
