"""Makes the Desktop and Start menu shortcuts. Run with this folder's .venv python.

The shortcut carries the same app ID the window uses, which is what lets
Windows pin the running window to the taskbar with this icon.
"""

import os
import sys

import pythoncom
from win32com.propsys import propsys, pscon
from win32com.shell import shell, shellcon

NAME = "Live Transcriber"
APP_ID = "Dylan.LiveTranscriber"
SCRIPT = "transcriber.py"
ICON = "transcriber.ico"

HERE = os.path.dirname(os.path.abspath(__file__))


def make(folder):
    link = pythoncom.CoCreateInstance(shell.CLSID_ShellLink, None,
                                      pythoncom.CLSCTX_INPROC_SERVER, shell.IID_IShellLink)
    link.SetPath(os.path.join(HERE, ".venv", "Scripts", "pythonw.exe"))
    link.SetArguments(f'"{os.path.join(HERE, SCRIPT)}"')
    link.SetWorkingDirectory(HERE)
    link.SetIconLocation(os.path.join(HERE, ICON), 0)
    link.SetDescription(NAME)
    store = link.QueryInterface(propsys.IID_IPropertyStore)
    store.SetValue(pscon.PKEY_AppUserModel_ID, propsys.PROPVARIANTType(APP_ID, pythoncom.VT_LPWSTR))
    store.Commit()
    path = os.path.join(folder, NAME + ".lnk")
    link.QueryInterface(pythoncom.IID_IPersistFile).Save(path, 0)
    print("made", path)


if __name__ == "__main__":
    make(shell.SHGetFolderPath(0, shellcon.CSIDL_DESKTOPDIRECTORY, None, 0))
    make(shell.SHGetFolderPath(0, shellcon.CSIDL_PROGRAMS, None, 0))
    sys.exit(0)
