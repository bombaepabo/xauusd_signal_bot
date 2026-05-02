# Use the specific version you requested
FROM python:3.11-slim

# Install uv binary directly from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy dependency files first to leverage Docker's layer caching
# This includes pyproject.toml, uv.lock, and requirements.txt if they exist
COPY pyproject.toml* uv.lock* requirements.txt* ./

# Install dependencies into the system site-packages
# This ensures numpy and other libs are globally accessible to the container
RUN if [ -f requirements.txt ]; then \
        uv pip install --system -r requirements.txt; \
    elif [ -f pyproject.toml ]; then \
        uv pip install --system .; \
    fi

# Copy the rest of your trading bot code
COPY . .

# Run as non-root for security
RUN useradd -m -u 1000 botuser && chown -R botuser /app
USER botuser

# Start the bot
CMD ["python", "webhook_runner.py"]