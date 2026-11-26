"""Entry point of the pre-compiled desktop app (PyInstaller).

No argument (double click)       -> desktop window, empty scene
Files only (open with / drag)    -> desktop window with these files
A tractviewer command first      -> regular CLI (tractviewer web|html|info|...)
--uninstall                      -> Linux AppImage: remove the menu entry and the installed copy
"""
import multiprocessing
import sys


def main() -> None:
    multiprocessing.freeze_support()
    from tractviewer.cli import COMMANDS, main as cli_main

    args = sys.argv[1:]
    # macOS passes -psn_0_xxx when launched from the Finder on old systems.
    args = [a for a in args if not a.startswith("-psn_")]
    if args == ["--uninstall"]:  # Linux AppImage menu entry, "Uninstall" action
        from tractviewer import linux_integration
        print(linux_integration.uninstall())
        return
    if not args or (args[0] not in COMMANDS and not args[0].startswith("-")):
        args = ["desktop", *args]
    sys.exit(cli_main(args))


if __name__ == "__main__":
    main()
