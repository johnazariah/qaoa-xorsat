FROM julia:1.12

WORKDIR /workspace

# Copy project files
COPY Project.toml Manifest.toml ./
COPY src/ src/
COPY scripts/ scripts/
COPY experiments/ experiments/
COPY test/ test/

# Install dependencies
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Verify tests pass
RUN julia --project=. -e 'using Pkg; Pkg.test()'

# Default: run the full table at p=11
ENTRYPOINT ["julia", "--project=.", "-t", "auto"]
CMD ["scripts/optimize_qaoa.jl"]
