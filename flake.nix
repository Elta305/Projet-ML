{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };
        python = pkgs.python311;
        pythonPackages = python.pkgs;
        devPkgs = with pkgs; [
          just
          python
        ];
        pythonPkgs = with pythonPackages; [
          numpy
          matplotlib
        ];
      in {
        app.default = {
        };
        devShells.default = pkgs.mkShell {
          buildInputs = devPkgs ++ pythonPkgs;
          shellHook = ''
          '';
        };
      }
    );
}
