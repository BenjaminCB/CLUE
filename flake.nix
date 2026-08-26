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
          python = pkgs.python3.withPackages (pythonPackages: [
            pythonPackages.beautifulsoup4
            # hypothesis drives the property-based tests in ./tests.
            pythonPackages.hypothesis
            pythonPackages.matplotlib
            # mypy lives in the same environment as the project's dependencies so
            # that it can resolve sympy/numpy instead of reporting import-not-found.
            pythonPackages.mypy
            pythonPackages.natsort
            pythonPackages.numpy
            pythonPackages.pyparsing
            pythonPackages.pytest
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
            ];
          };
        }
      );
    };
}
