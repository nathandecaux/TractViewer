; Inno Setup script of the Windows installer.
;   iscc /DAppVersion=1.2.3 /DSourceDir=..\..\dist\TractViewer /DOutputDir=..\..\dist packaging\windows\installer.iss
; Per-user install (no admin rights): the auto-updater can run it silently.
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\TractViewer"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif

[Setup]
AppId={{8C1E3F2A-5B7D-4E39-9A61-2F0D7C4B9E15}
AppName=TractViewer
AppVersion={#AppVersion}
AppVerName=TractViewer {#AppVersion}
AppPublisher=TractViewer
AppPublisherURL=https://github.com/nathandecaux/TractViewer
AppUpdatesURL=https://github.com/nathandecaux/TractViewer/releases
DefaultDirName={localappdata}\Programs\TractViewer
DefaultGroupName=TractViewer
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=TractViewer-{#AppVersion}-windows-x64-setup
SetupIconFile=..\build\icon.ico
UninstallDisplayIcon={app}\TractViewer.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=force
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; Files of the previous version (libraries renamed between versions)
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\TractViewer"; Filename: "{app}\TractViewer.exe"
Name: "{group}\{cm:UninstallProgram,TractViewer}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\TractViewer"; Filename: "{app}\TractViewer.exe"; Tasks: desktopicon

[Run]
; Interactive install: optional launch at the end
Filename: "{app}\TractViewer.exe"; Description: "{cm:LaunchProgram,TractViewer}"; Flags: nowait postinstall skipifsilent
; Silent install (auto-update): relaunch the app
Filename: "{app}\TractViewer.exe"; Flags: nowait; Check: WizardSilent
