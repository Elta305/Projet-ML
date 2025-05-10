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
        python = pkgs.python313;
        pythonPackages = python.pkgs;
        devPkgs = with pkgs; [
          just
          typst
          python
          stdenv.cc.cc.lib
          libz
          (texlive.combine {
            inherit
              (texlive)
              scheme-full
              adjustbox
              collectbox
              tcolorbox
              pgf
              xetex
              ;
          })
        ];
        pythonPkgs = with pythonPackages; [
          uv
          tkinter
        ];
      in {
        app.default = {
        };
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = devPkgs ++ pythonPkgs;
          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"
            export MPLBACKEND="TkAgg"
          '';
        };
      }
    );
}
