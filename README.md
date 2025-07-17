# Simple Agent API

Welcome to the Simple Agent API: a robust, production-ready application for serving Agents as an API. It includes:
  * A **FastAPI server** for handling API requests.
  * A **PostgreSQL database** for storing Agent sessions, knowledge, and memories.
  * A set of **pre-built Agents** to use as a starting point.

For more information, checkout [Agno](https://agno.link/gh) and give it a ⭐️

## Quickstart

Follow these steps to get your Agent API up and running:

> Prerequisites: [docker desktop](https://www.docker.com/products/docker-desktop) should be installed and running.

### Clone the repo

```sh
git clone https://github.com/agno-agi/agent-api.git
cd agent-api
```

### Configure API keys

We use GPT 4.1 as the default model, please export the `OPENAI_API_KEY` environment variable to get started.

```sh
export OPENAI_API_KEY="YOUR_API_KEY_HERE"
```

> **Note**: You can use any model provider, just update the agents in the `/agents` folder.

### Start the application

Run the application using docker compose:

```sh
docker compose up -d
```

This command starts:
* The **FastAPI server**, running on [http://localhost:8000](http://localhost:8000).
* The **PostgreSQL database**, accessible on `localhost:5432`.

Once started, you can:
* Test the API at [http://localhost:8000/docs](http://localhost:8000/docs).

### Connect to Agno Playground or Agent UI

* Open the [Agno Playground](https://app.agno.com/playground).
* Add `http://localhost:8000` as a new endpoint. You can name it `Agent API` (or any name you prefer).
* Select your newly added endpoint and start chatting with your Agents.

https://github.com/user-attachments/assets/a0078ade-9fb7-4a03-a124-d5abcca6b562

### Stop the application

When you're done, stop the application using:

```sh
docker compose down
```

## Prebuilt Agents

The `/agents` folder contains pre-built agents that you can use as a starting point.
- Web Search Agent: A simple agent that can search the web.
- Agno Assist: An Agent that can help answer questions about Agno.
  - Important: Make sure to load the `agno_assist` [knowledge base](http://localhost:8000/docs#/Agents/load_agent_knowledge_v1_agents__agent_id__knowledge_load_post) before using this agent.
- Finance Agent: An agent that uses the YFinance API to get stock prices and financial data.

## Development Setup

To setup your local virtual environment:

### Install `uv`

We use `uv` for python environment and package management. Install it by following the the [`uv` documentation](https://docs.astral.sh/uv/#getting-started) or use the command below for unix-like systems:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create Virtual Environment & Install Dependencies

Run the `dev_setup.sh` script. This will create a virtual environment and install project dependencies:

```sh
./scripts/dev_setup.sh
```

### Activate Virtual Environment

Activate the created virtual environment:

```sh
source .venv/bin/activate
```

(On Windows, the command might differ, e.g., `.venv\Scripts\activate`)

## Managing Python Dependencies

If you need to add or update python dependencies:

### Modify pyproject.toml

Add or update your desired Python package dependencies in the `[dependencies]` section of the `pyproject.toml` file.

### Generate requirements.txt

The `requirements.txt` file is used to build the application image. After modifying `pyproject.toml`, regenerate `requirements.txt` using:

```sh
./scripts/generate_requirements.sh
```

To upgrade all existing dependencies to their latest compatible versions, run:

```sh
./scripts/generate_requirements.sh upgrade
```

### Rebuild Docker Images

Rebuild your Docker images to include the updated dependencies:

```sh
docker compose up -d --build
```

## Community & Support

Need help, have a question, or want to connect with the community?

* 📚 **[Read the Agno Docs](https://docs.agno.com)** for more in-depth information.
* 💬 **Chat with us on [Discord](https://agno.link/discord)** for live discussions.
* ❓ **Ask a question on [Discourse](https://agno.link/community)** for community support.
* 🐛 **[Report an Issue](https://github.com/agno-agi/agent-api/issues)** on GitHub if you find a bug or have a feature request.

## Running in Production

This repository includes a `Dockerfile` for building a production-ready container image of the application.

The general process to run in production is:

1. Update the `scripts/build_image.sh` file and set your IMAGE_NAME and IMAGE_TAG variables.
2. Build and push the image to your container registry:

```sh
./scripts/build_image.sh
```
3. Run in your cloud provider of choice.

### Detailed Steps

1. **Configure for Production**
  * Ensure your production environment variables (e.g., `OPENAI_API_KEY`, database connection strings) are securely managed. Most cloud providers offer a way to set these as environment variables for your deployed service.
  * Review the agent configurations in the `/agents` directory and ensure they are set up for your production needs (e.g., correct model versions, any production-specific settings).

2. **Build Your Production Docker Image**
  * Update the `scripts/build_image.sh` script to set your desired `IMAGE_NAME` and `IMAGE_TAG` (e.g., `your-repo/agent-api:v1.0.0`).
  * Run the script to build and push the image:

    ```sh
    ./scripts/build_image.sh
    ```

3. **Deploy to a Cloud Service**
  With your image in a registry, you can deploy it to various cloud services that support containerized applications. Some common options include:

  * **Serverless Container Platforms**:
    * **Google Cloud Run**: A fully managed platform that automatically scales your stateless containers. Ideal for HTTP-driven applications.
    * **AWS App Runner**: Similar to Cloud Run, AWS App Runner makes it easy to deploy containerized web applications and APIs at scale.
    * **Azure Container Apps**: Build and deploy modern apps and microservices using serverless containers.

  * **Container Orchestration Services**:
    * **Amazon Elastic Container Service (ECS)**: A highly scalable, high-performance container orchestration service that supports Docker containers. Often used with AWS Fargate for serverless compute or EC2 instances for more control.
    * **Google Kubernetes Engine (GKE)**: A managed Kubernetes service for deploying, managing, and scaling containerized applications using Google infrastructure.
    * **Azure Kubernetes Service (AKS)**: A managed Kubernetes service for deploying and managing containerized applications in Azure.

  * **Platform as a Service (PaaS) with Docker Support**
    * **Railway.app**: Offers a simple way to deploy applications from a Dockerfile. It handles infrastructure, scaling, and networking.
    * **Render**: Another platform that simplifies deploying Docker containers, databases, and static sites.
    * **Heroku**: While traditionally known for buildpacks, Heroku also supports deploying Docker containers.

  * **Specialized Platforms**:
    * **Modal**: A platform designed for running Python code (including web servers like FastAPI) in the cloud, often with a focus on batch jobs, scheduled functions, and model inference, but can also serve web endpoints.

  The specific deployment steps will vary depending on the chosen provider. Generally, you'll point the service to your container image in the registry and configure aspects like port mapping (the application runs on port 8000 by default inside the container), environment variables, scaling parameters, and any necessary database connections.

4. **Database Configuration**
  * The default `docker-compose.yml` sets up a PostgreSQL database for local development. In production, you will typically use a managed database service provided by your cloud provider (e.g., AWS RDS, Google Cloud SQL, Azure Database for PostgreSQL) for better reliability, scalability, and manageability.
  * Ensure your deployed application is configured with the correct database connection URL for your production database instance. This is usually set via an environment variables.
