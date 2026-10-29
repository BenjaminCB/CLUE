{
  description = "Development environment for the CLUE noise paper";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      supportedSystems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          # qiskit's setup.py monkey-patches setuptools-rust internals and hard-pins 1.12.0, while
          # nixpkgs ships a newer one; build qiskit (only) against the version it expects.
          python3 = pkgs.python3.override {
            packageOverrides = self: super: {
              qiskit = super.qiskit.overridePythonAttrs (old: {
                build-system = [
                  self.setuptools
                  (super.setuptools-rust.overridePythonAttrs (_: rec {
                    version = "1.12.0";
                    src = pkgs.fetchPypi {
                      pname = "setuptools_rust";
                      inherit version;
                      hash = "sha256-2UqT8Ml3UcFwFFZfB73DJL7kXTls0buoPY56+SuUXww=";
                    };
                  }))
                ];
              });
              # nixpkgs' jupyterlab wrapper exports the literal string '$out/share/jupyter/lab' as
              # JUPYTERLAB_DIR (unexpanded), so the lab assets are never found; point it at the store path.
              jupyterlab = super.jupyterlab.overridePythonAttrs (old: {
                makeWrapperArgs = [
                  "--set"
                  "JUPYTERLAB_DIR"
                  "${placeholder "out"}/share/jupyter/lab"
                ];
              });
            };
          };
          python = python3.withPackages (pythonPackages: [
            pythonPackages.beautifulsoup4
            # hypothesis drives the property-based tests in ./tests.
            pythonPackages.hypothesis
            # jupyterlab runs the notebooks, e.g. papers/noise/sweeps.ipynb for the sweep CSVs.
            pythonPackages.jupyterlab
            pythonPackages.matplotlib
            # mypy lives in the same environment as the project's dependencies so
            # that it can resolve sympy/numpy instead of reporting import-not-found.
            pythonPackages.mypy
            pythonPackages.natsort
            pythonPackages.numpy
            pythonPackages.pyparsing
            pythonPackages.pytest
            # qiskit builds and transpiles the circuits in papers/noise.
            pythonPackages.qiskit
            pythonPackages.requests
            pythonPackages.scipy
            pythonPackages.sympy
          ]);
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.just
              python
              # basedpyright-langserver backs the lsp-mcp server (see .mcp.json),
              # giving Claude Code access to real inferred types via the LSP.
              pkgs.basedpyright
              # nodejs provides npx, used to run lsp-mcp on demand.
              pkgs.nodejs
            ];
          };
        }
      );
    };
}
