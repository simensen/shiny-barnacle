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

        python = pkgs.python312;

        pythonPackages = python.pkgs;

        # Python dependencies
        pythonEnv = python.withPackages (ps: with ps; [
          # Core dependencies
          fastapi
          uvicorn
          httpx

          # Optional: for development/testing
          pytest
          black
          ruff
          mypy

          # Uvicorn extras for better performance
          uvloop
          httptools
          websockets
        ]);

      in
      {
        # Development shell
        devShells.default = pkgs.mkShell {
          name = "toolbridge-dev";

          buildInputs = [
            pythonEnv

            # Optional: useful CLI tools
            pkgs.curl
            pkgs.jq
          ];

          shellHook = ''
            echo "🚀 LLM Response Transformer Proxy Development Environment"
            echo ""
            echo "Available commands:"
            echo "  python toolbridge.py    - Start the proxy"
            echo "  pytest                  - Run tests"
            echo ""
            echo "Python: $(python --version)"
            echo "FastAPI, uvicorn, httpx ready"
            echo ""
          '';

          # Set Python to not write bytecode (cleaner dev experience)
          PYTHONDONTWRITEBYTECODE = "1";

          # Ensure reproducible builds
          PYTHONHASHSEED = "0";
        };

        # Package the proxy as a runnable application
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "toolbridge";
          version = "0.1.0";

          src = ./.;

          buildInputs = [ pythonEnv ];

          installPhase = ''
            mkdir -p $out/bin $out/lib
            cp *.py $out/lib/

            # Create wrapper script
            cat > $out/bin/toolbridge << EOF
            #!${pkgs.bash}/bin/bash
            exec ${pythonEnv}/bin/python $out/lib/toolbridge.py "\$@"
            EOF
            chmod +x $out/bin/toolbridge
          '';

          meta = with pkgs.lib; {
            description = "OpenAI-compatible proxy for LLM tool call transformation";
            license = licenses.mit;
            platforms = platforms.unix;
          };
        };

        # Quick run commands
        apps = {
          default = self.apps.${system}.toolbridge;

          toolbridge = {
            type = "app";
            program = "${self.packages.${system}.default}/bin/toolbridge";
          };
        };
      }
    );
}
