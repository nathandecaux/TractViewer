# Conservé pour rétro-compatibilité: import la classe depuis le package
from tractviewer.core import TractViewer

if __name__ == "__main__":
    # Exemple minimal: afficher aide CLI
    import sys
    from tractviewer.cli import main
    if len(sys.argv) == 1:
        print("Usage: python TractViewer.py <fichiers> --screenshot out.png")
    main(sys.argv[1:])
