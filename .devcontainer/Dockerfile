FROM mcr.microsoft.com/devcontainers/base:bookworm

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
        # PDF text extraction (for paper analysis)
        poppler-utils \
        python3-pip \
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

# Install Python PDF libraries
RUN pip3 install --break-system-packages --no-cache-dir pymupdf

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

# Switch to non-root user
USER vscode

# Install Julia via juliaup (official version manager)
RUN curl -fsSL https://install.julialang.org | sh -s -- --yes
ENV PATH="/home/vscode/.juliaup/bin:${PATH}"

# Pre-install common Julia packages into a shared depot
RUN julia -e 'using Pkg; Pkg.add(["Revise", "OhMyREPL", "BenchmarkTools"])'

WORKDIR /workspace
