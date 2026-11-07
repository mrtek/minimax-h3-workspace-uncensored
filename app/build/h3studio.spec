# PyInstaller spec: python -m PyInstaller app/build/h3studio.spec --distpath app/build/dist --workpath app/build/work
from pathlib import Path

APP = Path(SPECPATH).parent

a = Analysis(
    [str(APP / "main.py")],
    pathex=[str(APP)],
    datas=[(str(APP / "ui" / "index.html"), "ui"), (str(APP / "skills"), "skills"), (str(APP / "bin"), "bin")],
    hiddenimports=["webview.platforms.edgechromium", "webview.platforms.winforms", "clr", "certifi"],
    excludes=["tkinter", "matplotlib", "numpy", "PIL", "pandas", "scipy", "IPython"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="MinimaxH3Workspace",
    icon=str(APP / "build" / "icon.ico"),
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="MinimaxH3Workspace")
