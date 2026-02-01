{
  description = "LLM Response Transformer Proxy - OpenAI-compatible proxy for tool call transformation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          name = "toolbridge-dev";

          buildInputs = with pkgs; [
            python312
            uv
            curl
            jq
          ];

          shellHook = ''
            export UV_PYTHON="python3.12"
            export UV_PYTHON_PREFERENCE="only-system"
            export UV_PROJECT_ENVIRONMENT=".direnv/.venv"
            export UV_CACHE_DIR=".direnv/.cache/uv"

            # Create virtual environment with uv if it doesn't exist
            if [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
              echo "Creating virtual environment with uv..."
              uv venv $UV_PROJECT_ENVIRONMENT
            fi

            # Activate virtual environment
            source $UV_PROJECT_ENVIRONMENT/bin/activate

            # Install dependencies from pyproject.toml if it exists
            if [ -f "pyproject.toml" ]; then
              echo "Installing dependencies from pyproject.toml with uv..."
              uv pip install -e ".[dev]" 2>/dev/null || uv pip install -e .
            fi

            if [ -t 1 ]; then
              echo ""
              echo "Toolbridge Development Environment"
              echo ""
              echo "Python: $(python --version) ($(which python))"
              echo "UV: $(uv --version)"
              echo "Virtual environment: $VIRTUAL_ENV"
              echo ""
              echo "Quick start:"
              echo "  make run        Run the proxy"
              echo "  make test       Run tests"
              echo "  make check      Run all checks"
              echo "  make help       Show all commands"
              echo ""
            fi
          '';

          PYTHONDONTWRITEBYTECODE = "1";
          PYTHONHASHSEED = "0";
        };
      }
    );
}
