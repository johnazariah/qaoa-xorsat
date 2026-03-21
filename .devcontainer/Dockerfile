FROM mcr.microsoft.com/devcontainers/base:bookworm

# Julia version
ARG JULIA_VERSION=1.11.4

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        wget \
        ca-certificates \
        gnupg \
        lsb-release \
        # LaTeX (for paper writing / figures)
        texlive-base \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-science \
        texlive-bibtex-extra \
        latexmk \
        biber \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update && apt-get install -y gh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Julia
RUN ARCH=$(dpkg --print-architecture) \
    && if [ "$ARCH" = "arm64" ]; then JULIA_ARCH="aarch64"; else JULIA_ARCH="x86_64"; fi \
    && JULIA_MINOR=$(echo ${JULIA_VERSION} | cut -d. -f1-2) \
    && curl -fsSL "https://julialang-s3.julialang.org/bin/linux/${JULIA_ARCH}/${JULIA_MINOR}/julia-${JULIA_VERSION}-linux-${JULIA_ARCH}.tar.gz" \
        | tar -xz -C /opt \
    && ln -s /opt/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia

# Switch to non-root user
USER vscode

# Pre-install common Julia packages into a shared depot
RUN julia -e 'using Pkg; Pkg.add(["Revise", "OhMyREPL", "BenchmarkTools"])'

WORKDIR /workspace
