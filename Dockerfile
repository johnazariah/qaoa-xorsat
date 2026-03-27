FROM --platform=$BUILDPLATFORM julia:1.12

# Support multi-arch: amd64 (Azure/GCE) and arm64 (Apple Silicon, Ampere)
ARG TARGETPLATFORM
ARG BUILDPLATFORM

WORKDIR /workspace

# Copy project files
COPY Project.toml Manifest.toml ./
COPY src/ src/
COPY scripts/ scripts/
COPY experiments/ experiments/
COPY test/ test/

# Install dependencies and precompile
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Verify tests pass
RUN julia --project=. -e 'using Pkg; Pkg.test()'

# Results volume mount point
VOLUME /workspace/results

# Default: run with all available threads
ENTRYPOINT ["julia", "--project=.", "-t", "auto"]
CMD ["scripts/optimize_qaoa.jl"]
