; Inno Setup script: ISCC.exe app\build\h3studio.iss  (after the PyInstaller build)
#define AppName "Minimax H3 Workspace"
#define AppVersion "1.1.9"
#define AppExe "MinimaxH3Workspace.exe"

[Setup]
AppId={{6B0E3C1E-5D7A-4C64-9A0B-7E2B1C9F4A31}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppName}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; always show the folder page (by default Inno skips it on a re-install and silently reuses the old folder)
DisableDirPage=no
; per-user install: no admin rights needed, and the app can write next to itself
PrivilegesRequired=lowest
OutputDir=output
OutputBaseFilename=Minimax.H3.Workspace.Uncensored-Setup_v1.1.9
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; the maker's own settings, library and keys live in dist\...\data: never ship or overwrite them
Source: "dist\MinimaxH3Workspace\*"; DestDir: "{app}"; Excludes: "data,data\*,data_location.txt"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[UninstallDelete]
; the app's own small files (settings, library list, chats): removed together with the program
Type: filesandordirs; Name: "{app}\data"
Type: files; Name: "{app}\data_location.txt"

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
; an update started from inside the app runs silently: open the app again when it is done
Filename: "{app}\{#AppExe}"; Flags: nowait skipifnotsilent

; ComfyUI and models live in the folder chosen in the app, outside {app}, and are kept on uninstall on purpose:
; they are tens of gigabytes. Settings go to <that folder>\data (see datadir.py).
