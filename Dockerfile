# ── Stage 1: Build dependencies with uv ──────────────────────────────────────
FROM python:3.11-slim AS builder

# Install uv binary directly from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation for slightly faster startup
ENV UV_COMPILE_BYTECODE=1

# Copy ONLY dependency files to leverage Docker's layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
# We use --frozen to ensure the container matches your uv.lock exactly
RUN uv sync --frozen --no-dev --no-install-project

# ── Stage 2: Lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Add the virtual environment to the PATH so 'python' points to the right place
ENV PATH="/app/.venv/bin:$PATH"

# Copy ONLY the files this specific bot needs
COPY webhook_runner.py .
COPY signal_engine.py  .
COPY discord_sender.py .
COPY config.py         .

# Run as non-root for security
RUN useradd -m -u 1000 botuser
USER botuser

# Health check - verifies the SignalEngine class can be imported[cite: 1]
HEALTHCHECK --interval=60s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "from signal_engine import SignalEngine; print('healthy')"

# Start the bot[cite: 1]
CMD ["python", "webhook_runner.py"]